import matplotlib.pyplot as plt
import numpy as np
from philanthropy.visualisation import plot_affinity_distribution, plot_retention_waterfall

def test_plot_affinity_distribution():
    scores = np.random.uniform(0, 100, 100)
    
    # Test without labels
    ax = plot_affinity_distribution(scores)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Affinity Score Distribution"
    
def test_plot_affinity_distribution_with_labels():
    scores = np.random.uniform(0, 100, 100)
    labels = np.random.randint(0, 2, 100)
    
    ax = plot_affinity_distribution(scores, labels=labels)
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Affinity Score Distribution by Major-Gift Label"

def test_plot_retention_waterfall():
    ax = plot_retention_waterfall(
        starting_donors=1000, 
        acquired=200, 
        lapsed=150, 
        recovered=50
    )
    assert isinstance(ax, plt.Axes)
    assert ax.get_title() == "Donor Retention Waterfall"
    assert len(ax.patches) == 5  # 5 bars
