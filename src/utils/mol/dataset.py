CACHE_GRAPH = False
CACHE_MOL = False

def cache_graph() -> bool:
    return CACHE_GRAPH

def cache_mol() -> bool:
    return CACHE_MOL

def set_cache_graph(cache_graph: bool) -> None:
    global CACHE_GRAPH
    CACHE_GRAPH = cache_graph

def set_cache_mol(cache_mol: bool) -> None:
    global CACHE_MOL
    CACHE_MOL = cache_mol

def empty_cache():
    pass
