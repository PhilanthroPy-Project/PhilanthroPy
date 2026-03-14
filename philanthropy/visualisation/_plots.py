import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_affinity_distribution(scores, labels=None) -> plt.Axes:
    """
    Plots a Seaborn KDE or histogram of the 0-100 affinity scores. 
    If actual major-gift labels are provided, plots overlaid distributions 
    (Major vs. Non-Major) to show model separation.
    
    Parameters
    ----------
    scores : array-like
        The generated affinity scores (0-100 scale).
    labels : array-like, optional
        The true binary labels for the donors (0=Non-Major, 1=Major).
    
    Returns
    -------
    matplotlib.axes.Axes
        The underlying axes object for further customization.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if labels is not None:
        df = pd.DataFrame({'Score': scores, 'Label': labels})
        # Map labels to human-readable strings for the legend
        df['Label'] = df['Label'].map({0: 'Non-Major', 1: 'Major'})
        sns.histplot(
            data=df, 
            x='Score', 
            hue='Label', 
            kde=True, 
            bins=20, 
            alpha=0.6, 
            ax=ax
        )
        ax.set_title("Affinity Score Distribution by Major-Gift Label")
    else:
        sns.histplot(
            x=scores, 
            kde=True, 
            bins=20, 
            alpha=0.6, 
            ax=ax
        )
        ax.set_title("Affinity Score Distribution")
        
    ax.set_xlabel("Affinity Score (0-100)")
    ax.set_ylabel("Frequency")
    
    return ax

def plot_retention_waterfall(starting_donors, acquired, lapsed, recovered) -> plt.Axes:
    """
    Generates a step-by-step waterfall chart showing the net change 
    in the donor file year-over-year.

    Parameters
    ----------
    starting_donors : int
        The number of donors at the beginning of the period.
    acquired : int
        The number of newly acquired donors.
    lapsed : int
        The number of donors who lapsed (entered as a positive integer).
    recovered : int
        The number of previously lapsed donors who recovered.

    Returns
    -------
    matplotlib.axes.Axes
        The underlying axes object for further customization.
    """
    categories = ['Starting', 'Acquired', 'Lapsed', 'Recovered', 'Ending']
    
    # Lapsed acts as a negative flow
    ending_donors = starting_donors + acquired - lapsed + recovered
    values = [starting_donors, acquired, -lapsed, recovered, ending_donors]
    
    colors = ['gray', 'green', 'red', 'blue', 'black']
    
    # Calculate bottom positions for the bars
    bottoms = [
        0, 
        starting_donors, 
        starting_donors + acquired, 
        starting_donors + acquired - lapsed, 
        0
    ]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(categories, values, bottom=bottoms, color=colors, edgecolor='black')
    
    ax.set_title("Donor Retention Waterfall")
    ax.set_ylabel("Number of Donors")
    
    # Add data labels
    for i, (cat, val, bot) in enumerate(zip(categories, values, bottoms)):
        # Calculate the Y position to center the text on the bar
        y_pos = bot + val / 2.0
        ax.text(i, y_pos, str(abs(val)), ha='center', va='center', color='white', fontweight='bold')
        
    return ax
