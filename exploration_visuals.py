import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def make_hist_scatter(df, feature, bins=10, norm_attendance=True):
    """
    This function makes two plots:
    Left: Histogram of the specified feature.
    Right: Scatter plot of attendance vs. the specified feature.
    
    This function should be used for continuous numerical features.
    For discrete numerical features or categorical features, use make_bar_scatter (below).
    
    Inputs:
    df: Pandas DataFrame that contains the data.
    feature: The name of the column we want to look at.
    bins: Number of bins to use for histogram.
    norm_attendance: Whether or not to use attendance that has been divided by stadium capacity.
    True: Use att_div_capacity
    False: Use attendance
    """
    fig, ax = plt.subplots(ncols=2, figsize=(15,7))
    
    # Histogram
    df[feature].plot.hist(bins=bins,
                          ax=ax[0],
                          xlabel=feature,
                          title=f"Histogram of {feature}")
    
    # Scatter plot
    if norm_attendance:
        att_name = 'att_div_capacity'
        att_label = 'attendance / capacity'
    else:
        att_name = 'attendance'
        att_label = 'attendance'
    df.plot.scatter(x=feature,
                    y=att_name,
                    ax=ax[1],
                    xlabel=feature,
                    ylabel=att_label,
                    title=f"{att_label} vs. {feature}",
                    alpha=0.5)
    
    fig.tight_layout()
    
def make_bar_scatter(df, feature, norm_attendance=True):
    """
    This function makes two plots:
    Left: Bar chart of the specified feature.
    Right: Scatter plot of attendance vs. the specified feature.
    
    This function should be used for discrete numerical features and categorical features.
    For continuous numerical features, use make_hist_scatter (above).
    
    Inputs:
    df: Pandas DataFrame that contains the data.
    feature: The name of the column we want to look at.
    norm_attendance: Whether or not to use attendance that has been divided by stadium capacity.
    True: Use att_div_capacity
    False: Use attendance
    """
    fig, ax = plt.subplots(ncols=2, figsize=(15,7))
    
    # Bar chart
    df.groupby(by=feature).agg({feature:'count'}).plot.bar(ax=ax[0])
    
    # Scatter plot
    if norm_attendance:
        att_name = 'att_div_capacity'
        att_label = 'attendance / capacity'
    else:
        att_name = 'attendance'
        att_label = 'attendance'
    df.plot.scatter(x=feature,
                    y=att_name,
                    ax=ax[1],
                    xlabel=feature,
                    ylabel=att_label,
                    title=f"{att_label} vs. {feature}",
                    alpha=0.5)
    
    df.groupby(feature).agg({feature:'last',att_name:'mean'}).plot.scatter(x=feature,
                                                                           y=att_name,
                                                                           ax=ax[1],
                                                                           color='darkred',
                                                                           alpha=1,
                                                                           s=250,
                                                                           marker="_")
    
    fig.tight_layout()
    
def attendance_histograms(df, feature, norm_attendance=True, label_dict=None):
    """
    This function makes histograms of attendance vs. a specified feature.
    Inputs:
    df: Pandas DataFrame that includes the data.
    feature: The feature to use to make separate histograms.
    norm_attendance: If True, use att_div_capacity. If False, use attendance.
    label_dict: If specified, it uses the label_dict for subplot titles.
    """
    # Find the data type of the feature
    feat_type = df[feature].dtype
    
    # Get unique values of the feature. Sort it if numeric.
    if feat_type == 'object':
        feat_uniq = df[feature].unique()
    else:
        feat_uniq = np.sort(df[feature].unique())
    
    fig, ax = plt.subplots(ncols=3, nrows = int(np.ceil(len(feat_uniq)/3)), figsize=(15,5*int(np.ceil(len(feat_uniq)/3))))
    
    for i, val in enumerate(feat_uniq):
        df[df[feature]==val].plot(y='att_div_capacity', kind='hist', ax=ax[i//3, i%3])
        if label_dict == None:
            ax[i//3,i%3].set_title(val)
        else:
            ax[i//3,i%3].set_title(label_dict[i])
            
    fig.tight_layout()
    
def make_scatter_plots(df, feature, split_by_team=False, label_dict=None):
    """
    This function makes scatter plots of attendance vs. a specified feature.
    Inputs:
    df: Pandas DataFrame that includes the data
    feature: The feature to use on the x-axis of scatter plots
    split_by_team: If True, it makes separate plots for each team as well as all data together.
    """
    if split_by_team:
        fig, ax = plt.subplots(nrows=10, ncols=3, figsize=(15,50))
        
        # Make separate plots for each team
        for i in np.sort(df['home_team'].unique()):
            r = i // 3
            c = i % 3
            df[df['home_team']==i].plot(x=feature,
                                        y='att_div_capacity',
                                        kind='scatter',
                                        ax=ax[r,c],
                                        alpha=0.5)
            
            ax[r,c].set_xlabel(feature)
            ax[r,c].set_ylabel('Normalized Attendance')
            if label_dict == None:
                ax[r,c].set_title(i)
            else:
                ax[r,c].set_title(label_dict[i])
        
        # Make plot combining all the data
        r = (i+1) // 3
        c = (i+1) % 3
        df.plot(x=feature,
                y='att_div_capacity',
                kind='scatter',
                ax=ax[r,c],
                alpha=0.5)
        
        ax[r,c].set_xlabel(feature)
        ax[r,c].set_ylabel('Normalized Attendance')
        ax[r,c].set_title('All')
        
        fig.tight_layout()
            
    else:
        fig, ax = plt.subplots(figsize=(8,8))
        df.plot(x=feature,
                y='att_div_capacity',
                kind='scatter',
                ax=ax,
                alpha=0.5)
        
        ax.set_xlabel(feature)
        ax.set_ylabel('Normalized Attendance')
        ax.set_title('All')