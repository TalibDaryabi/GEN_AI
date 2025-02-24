from graphviz import Digraph

from Projects.ai_news.graph import create_graph


def show_graph(graph):
    """
    Visualize the workflow graph using Graphviz.

    Args:
        graph (StateGraph): The Langraph StateGraph object to visualize.
    """
    # Create the graph
    graph = create_graph()

    # Initialize a directed graph
    dot = Digraph(comment='Workflow Graph')

    # Add nodes to the graph
    for node in graph.nodes:
        dot.node(node, node)  # Use node name as both ID and label

    # Add edges to the graph
    for from_node, to_node in graph.edges:
        dot.edge(from_node, to_node)

    # Render the graph to a file and open it
    try:
        dot.render('workflow_graph', format='pdf', view=True)
    except Exception as e:
        print(f"Error rendering graph: {e}. Ensure Graphviz is installed.")\

if __name__ == "__main__":
    show_graph(create_graph())
