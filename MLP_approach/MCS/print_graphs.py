import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def render_human(self):
    if self.render_mode != "human":
        return

    if not hasattr(self, "nodes_artist"):
        self.ax.clear()
        self.fig.subplots_adjust(left=0.30)

        labels = {
            n: f"{n}\n(d={self.G.nodes[n]['duration']})"
            for n in self.G.nodes
        }

        self.nodes_artist = nx.draw_networkx_nodes(
            self.G,
            self.pos,
            ax=self.ax,
            node_size=500
        )

        self.edges_artist = nx.draw_networkx_edges(
            self.G,
            self.pos,
            ax=self.ax,
            arrows=True
        )

        self.labels_artist = nx.draw_networkx_labels(
            self.G,
            self.pos,
            labels=labels,
            font_size=7,
            ax=self.ax
        )

        self.seed_text = self.ax.text(
            -0.28,
            0.97,
            "",
            transform=self.ax.transAxes,
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                facecolor='white',
                alpha=0.9,
                edgecolor='black'
            )
        )

        self.cpu_title_text = self.ax.text(
            -0.28,
            0.91,
            "CPU State",
            transform=self.ax.transAxes,
            fontsize=9,
            fontweight="bold",
            bbox=dict(
                facecolor='white',
                alpha=0.9,
                edgecolor='black'
            )
        )

        self.cpu_texts = {}

        for cpu_id in self.env['cpu_state']:

            txt = self.ax.text(
                -0.28,
                0.86 - (cpu_id * 0.055),
                "",
                transform=self.ax.transAxes,
                fontsize=8,
                bbox=dict(
                    facecolor='white',
                    alpha=0.8,
                    edgecolor='gray'
                )
            )

            self.cpu_texts[cpu_id] = txt

    node_colors = []

    for n in self.G.nodes:
        status = self.env['graph_tasks'][n]["status"]
        if status == "completed":
            node_colors.append("#2ECC71")
        elif status == "in_progress":
            node_colors.append("#F39C12")
        else:
            node_colors.append("#BDC3C7")

    self.nodes_artist.set_color(node_colors)

    if self.is_done:
        self.ax.set_title(
            f"All tasks completed in {self.total_time} time units!",
            fontsize=12
        )
    else:
        self.ax.set_title(
            f"Time: {self.total_time}",
            fontsize=12
        )


    self.seed_text.set_text(f"Seed: {self.seed}")

    for cpu_id, state in self.env['cpu_state'].items():
        if state['current_task'] is not None:
            text = (
                f"CPU {cpu_id}: "
                f"Task {state['current_task']}"
            )
        else:
            text = f"CPU {cpu_id}: Idle"

        self.cpu_texts[cpu_id].set_text(text)

    plt.draw()
    plt.pause(0.1)