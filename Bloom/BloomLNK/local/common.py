from typing import List

import networkx as nx


def create_graph_from_dags(
    dags: List[dict],
    col_name_to_node_type: dict,
    col_name_to_node_properties: dict,
    col_name_to_edge_type: dict,
    col_name_to_edge_properties: dict,
):
    G = nx.Graph()
    for dag in dags:
        # add nodes
        for cname in col_name_to_node_type:
            db_node_id = dag.get(cname)
            if db_node_id is not None:
                node_type = col_name_to_node_type[cname]
                node_id = (node_type, db_node_id)
                if node_id not in G.nodes:
                    G.add_node(node_id, node_type=node_type, properties={})
                # add properties
                if cname in col_name_to_node_properties:
                    properties = G.nodes[node_id]["properties"]
                    for p in col_name_to_node_properties[cname]:
                        prop = dag.get(p)
                        if prop is not None:
                            properties[p] = prop
        # add edges
        for cname_1, cname_2 in col_name_to_edge_type:
            db_node_id_1 = dag.get(cname_1)
            db_node_id_2 = dag.get(cname_2)
            if db_node_id_1 is not None and db_node_id_2 is not None:
                node_type_1 = col_name_to_node_type[cname_1]
                node_type_2 = col_name_to_node_type[cname_2]
                node_id_1 = (node_type_1, db_node_id_1)
                node_id_2 = (node_type_2, db_node_id_2)
                edge_type = col_name_to_edge_type[(cname_1, cname_2)]
                if G.has_edge(node_id_1, node_id_2) == False:
                    G.add_edge(
                        node_id_1,
                        node_id_2,
                        edge_type=edge_type,
                        properties={},
                    )
                # add properties
                if (cname_1, cname_2) in col_name_to_edge_properties:
                    properties = G[node_id_1][node_id_2]["properties"]
                    for p in col_name_to_edge_properties[(cname_1, cname_2)]:
                        prop = dag.get(p)
                        if prop is not None:
                            properties[p] = prop
    return G


def get_edge_subgraph(G: nx.Graph, edge_types: List[str]):
    keep = []
    for n1, n2, e in G.edges(data=True):
        if e["edge_type"] in edge_types:
            keep.append((n1, n2))
    return G.edge_subgraph(keep)


def get_node_subgraph(G: nx.Graph, node_types: List[str]):
    keep = []
    for n in G.nodes:
        if n[0] in node_types:
            keep.append(n)
    return G.subgraph(keep)
