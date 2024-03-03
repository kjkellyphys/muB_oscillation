import os
import os.path
import collections
import numpy as np


class memodict_(collections.OrderedDict):
    def __init__(self, f, maxsize=1):
        collections.OrderedDict.__init__(self)
        self.f = f
        self.maxsize = maxsize

    def __getitem__(self, key, extra=None):
        if super().__contains__(key):
            return super().__getitem__(key)
        if self.maxsize == 0:
            return self.f(key, extra)
        if len(self) == self.maxsize:
            self.popitem(last=False)
        ret = self.f(key, extra)
        super().__setitem__(key, ret)
        return ret

    def __call__(self, key, extra):
        return self.__getitem__(key, extra)


def memodict(f, maxsize=1):
    """Memoization decorator for a function taking a single argument"""
    m = memodict_(f, maxsize)
    return m


class store_entry:
    def __init__(self, name, dependents, atomic_operation, override_params=None):
        self.name = name
        if dependents is None:
            dependents = []
        self.dependents = dependents
        self.atomic_operation = atomic_operation
        self.override_params = override_params

    def __call__(self, *args):
        return self.atomic_operation(*args)


def try_except(success, failure, *exceptions):
    try:
        return success()
    except exceptions or Exception:
        return failure() if callable(failure) else failure


def pretty_type(v):
    if type(v) is np.ndarray:
        return "<ndarray " + repr(v.dtype) + ", " + repr(v.shape) + ">"
    elif type(v) is list:
        return "[" + ", ".join([pretty_type(vv) for vv in v]) + "]"
    elif type(v) is tuple:
        return "(" + ", ".join([pretty_type(vv) for vv in v]) + ")"
    else:
        return repr(type(v))


class store_initialized_entry(store_entry):
    def __init__(
        self, uninitialized, the_store, physical_props, props, cache_size=None
    ):
        store_entry.__init__(
            self,
            uninitialized.name,
            uninitialized.dependents,
            uninitialized.atomic_operation,
            override_params=uninitialized.override_params,
        )

        self.the_store = the_store

        if cache_size is None:
            cache_size = 1
        self.cache_size = cache_size

        self.implicit_physical_props = []
        self.physical_props = []
        self.props = []
        self.physical_props_indices = []
        self.props_indices = []
        self.is_physical = []
        for i, name in enumerate(self.dependents):
            if name in physical_props:
                self.is_physical.append(True)
                self.physical_props.append(name)
                self.physical_props_indices.append(i)
            elif name in props:
                self.is_physical.append(False)
                self.props.append(name)
                self.props_indices.append(i)
            else:
                raise ValueError("Not in props or physical_props:", name)

        self.cache = memodict(self.compute, self.cache_size)

    def compute(self, parameters, physical_parameters):
        values = [None for i in range(len(self.dependents))]
        for v, i in zip(parameters, self.physical_props_indices):
            values[i] = v
        for prop, i in zip(self.props, self.props_indices):
            values[i] = self.the_store.get_prop(prop, physical_parameters)
        return self.atomic_operation(*values)

    def __call__(self, physical_parameters):
        if self.override_params is not None:
            physical_parameters = physical_parameters.copy()
            physical_parameters.update(self.override_params)
        these_params = tuple([physical_parameters[k] for k in self.physical_props])
        these_params += tuple(
            [physical_parameters[k] for k in self.implicit_physical_props]
        )
        res = self.cache(these_params, physical_parameters)
        return res


OverrideParams = collections.namedtuple(
    "OverrideParams", ["params", "values"], defaults=[tuple(), tuple()]
)
ComputeNode = collections.namedtuple(
    "ComputeNode",
    ["name", "override_params", "is_physical", "dependents"],
    defaults=[OverrideParams(), False, tuple()],
)
ComputeNodeKey = collections.namedtuple(
    "ComputeNodeKey", ["name", "override_params"], defaults=[OverrideParams()]
)
GroupKey = collections.namedtuple(
    "ComputeNodeKey",
    ["physical_params", "override_params"],
    defaults=[OverrideParams()],
)


def get_node_key(node):
    return ComputeNodeKey(*node[:2])


def make_override_key(override_params):
    if override_params is None:
        override_params = {}
    elif type(override_params) is OverrideParams:
        override_params = OverrideParams._make(override_params)
    elif type(override_params) is dict:
        override_keys = tuple(sorted(override_params.keys()))
        override_vals = tuple([override_params[k] for k in override_keys])
        override_params = (override_keys, override_vals)
    return override_params


def make_override_dict(override_params):
    if override_params is None:
        override_params = {}
    elif type(override_params) is OverrideParams:
        override_params = override_params._asdict()
    elif type(override_params) is dict:
        override_params = dict(override_params)
    return override_params


def update_override(current_op, context):
    context = make_override_dict(context)
    current_op = make_override_dict(current_op)
    context.update(current_op)
    return make_override_key(context)


class store:
    def __init__(self, default_cache_size=1):
        self.default_cache_size = default_cache_size
        self.props = dict()
        self.cache_sizes = dict()
        self.initialized_props = dict()

    def get_prop(self, name, physical_parameters=None):
        if physical_parameters is None:
            physical_parameters = dict()
        return self.initialized_props[name](physical_parameters)

    def call(self, name, physical_parameters=None):
        if physical_parameters is None:
            physical_parameters = dict()

        context = make_override_key(physical_parameters)
        to_process_stack = [(name, context)]

        visited_groups = {}
        groups = []
        visited_nodes = {}
        while len(to_process_stack) > 0:
            name, context = to_process_stack.pop()
            if name in self.initialized_props:
                is_physical = False
                current = self.initialized_props[name]
                override_key = update_override(current.override_params, context)
                physical_deps = tuple(
                    sorted(
                        set(current.physical_props + current.implicit_physical_props)
                    )
                )

                group_key = GroupKey(physical_deps, override_key)
                node_key = ComputeNodeKey(name, override_key)

                if group_key not in visited_groups:
                    visited_groups[group_key] = [len(visited_groups), []]
                    groups.append(group_key)

                if node_key not in visited_nodes:
                    visited_groups[group_key][1].append(node_key)
                    dependents = []
                    for dep_name in current.dependents:
                        to_process_stack.append((dep_name, override_key))

                        dep = self.initialized_props[dep_name]
                        dep_override_key = update_override(
                            dep.override_params, override_key
                        )
                        dep_node_key = ComputeNodeKey(dep_name, dep_override_key)

                        dependents.append(dep_node_key)
                    node = ComputeNode(name, override_key, is_physical, dependents)
                    visited_nodes[node_key] = node
            else:
                is_physical = True
                override_key = update_override(None, context)
                physical_deps = tuple()
                dependents = []
                node = ComputeNode(name, override_key, is_physical, dependents)
                if node_key not in visited_nodes:
                    visited_nodes[node_key] = node

        node_counts = [len(visited_groups[group_key]) for group in groups]
        node_keys = [
            node_key
            for group_key in groups
            for node_key in visited_groups[group_key][1]
        ]
        node_ids = dict(node_keys, range(len(node_keys)))
        nodes = [visited_nodes[nk] for nk in node_keys]
        n_deps = [len(node.dependents) for nk, node in zip(node_keys, nodes)]
        parameter_node_ids = [
            [
                (node_ids[dep_key], i)
                for i, dep_key in enumerate(node.dependents)
                if not visited_nodes[dep_key].is_physical
            ]
            for nk, node in zip(node_keys, nodes)
        ]
        physical_param_names = [
            [
                (dep_key.name, i)
                for i, dep_key in node.dependents
                if visited_nodes[dep_key].is_physical
                and (dep_key.name not in visited_nodes[dep_key].override_params.params)
            ]
            for nk, node in zip(node_keys, nodes)
        ]
        override_param_names = [
            [
                (
                    dep_key.name,
                    make_override_dict(dep_key.override_params)[dep_key.name],
                    i,
                )
                for dep_key in node.dependents
                if visited_nodes[dep_key].is_physical
                and (dep_key.name in visited_nodes[dep_key].override_params.params)
            ]
            for nk, node in zip(node_keys, nodes)
        ]
        atomic_ops = [
            self.initialized_props[node.name].atomic_operation for node in nodes
        ]

        def make_group_eval(group_i):
            group_key = groups[group_i]
            _, group_node_keys = visited_groups[group_key]
            cache_size = np.sum(node_counts[group_i + 1 :])
            return_size = np.sum(node_counts[group_i])
            n_above = np.sum(node_counts[: group_i + 1])

            node_idx = tuple(reversed([tuple(node_ids[nk]) for nk in group_node_keys]))
            parameter_dep_idxs = tuple([tuple(parameter_node_ids[i]) for i in node_idx])
            cache_idxs = tuple(
                [tuple([x - n_above for x in parameter_dep_ids[i]]) for i in node_idx]
            )
            physical_dep_idxs = tuple(
                [tuple(physical_param_names[i]) for i in node_idx]
            )
            override_dep_idxs = tuple(
                [tuple(override_param_names[i]) for i in node_idx]
            )
            atomic = tuple([tuple(atomic_ops[i]) for i in node_idx])

            n_dependents = tuple([n_deps[i] for i in node_idx])

            def group_eval(physical_parameters, cache):
                nodes = [None for i in range(return_size)] + list(cache)
                for i in range(return_size):
                    n = n_dependents[i]
                    vals = [None for i in range(n_dependents[i])]
                    for cache_i, arg_i in cache_idxs[i]:
                        vals[arg_i] = cache_i
                    for p_name, arg_i in physical_dep_idxs:
                        vals[arg_i] = physical_parameters[p_name]
                    for o_name, o_val, arg_i in override_dep_idxs:
                        vals[arg_i] = o_val
                    nodes[i] = atomic[i](*vals)
                return nodes[:, return_size]

            return group_eval

        # Now grouped up
        group_funs = []
        for group_i, group_key in enumerate(groups):
            fun = make_group_eval(group_i)
            group_funs.append(fun)

    def add_prop(
        self, name, dependents, atomic_operation, cache_size=None, override_params=None
    ):
        prop = store_entry(
            name, dependents, atomic_operation, override_params=override_params
        )
        self.props[name] = prop
        self.cache_sizes[name] = cache_size

    def reset_cache(self, props):
        for prop in props:
            self.initialized_props[prop].clear()

    def initialize(self, keep_cache=False):
        old_initialized_props = self.initialized_props
        if not keep_cache:
            del old_initialized_props
        self.initialized_props = dict()
        props = self.props.keys()
        dependents = set()
        for prop in props:
            deps = self.props[prop].dependents
            if deps is not None:
                dependents.update(deps)
        props = set(props)
        physical_props = dependents - props
        for prop in props:
            entry = self.props[prop]
            if entry.dependents is not None:
                these_deps = set(entry.dependents)
            else:
                these_deps = set()
            these_physical_props = these_deps - props
            these_props = these_deps - these_physical_props
            initialized_entry = store_initialized_entry(
                entry,
                self,
                these_physical_props,
                these_props,
                cache_size=self.cache_sizes[prop],
            )
            self.initialized_props[prop] = initialized_entry
        if keep_cache:
            keys = old_initialized_props.keys()
            for k in keys:
                if k in self.initialized_props:
                    self.initialized_props[k].cache = old_initialized_props[k].cache
            del old_initialized_props

        def add_implicit_dependencies(prop):
            initialized_entry = self.initialized_props[prop]
            if len(initialized_entry.implicit_physical_props) == 0:
                prop_deps = set()
                for dprop in initialized_entry.props:
                    add_implicit_dependencies(dprop)
                    prop_deps |= set(self.initialized_props[dprop].physical_props)
                    prop_deps |= set(
                        self.initialized_props[dprop].implicit_physical_props
                    )
                prop_deps -= set(initialized_entry.physical_props)
                if initialized_entry.override_params is not None:
                    prop_deps -= set(initialized_entry.override_params.keys())
                initialized_entry.implicit_physical_props = list(prop_deps)

        for prop in props:
            add_implicit_dependencies(prop)


class store_view:
    def __init__(self, the_store, parameters):
        self.the_store = the_store
        self.parameters = parameters

    def __getitem__(self, prop_name):
        return self.the_store.get_prop(prop_name, self.parameters)
