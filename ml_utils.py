import pandas as pd
import numpy as np
import math
import sklearn.datasets
import ipywidgets as widgets

##Seaborn for fancy plots. 
#%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams["figure.figsize"] = (8,8)

from distfit import distfit


class edaDF:
    """
    A class used to perform common EDA tasks

    ...

    Attributes
    ----------
    data : dataframe
        a dataframe on which the EDA will be performed
    target : str
        the name of the target column
    cat : list
        a list of the names of the categorical columns
    num : list
        a list of the names of the numerical columns

    Methods
    -------
    """
    def __init__(self, data, target):
        """
        Initialize an edaDF object.

        Parameters
        ----------
        data : pandas.DataFrame
            The dataset on which exploratory data analysis will be performed.
        target : str
            The name of the target column in the dataset.

        Attributes
        ----------
        data : pandas.DataFrame
            The full dataset.
        target : str
            The target column name.
        cat : list
            A list of categorical column names (initially empty).
        num : list
            A list of numerical column names (initially empty).
        """

        self.data = data
        self.target = target
        self.col = []
        self.cat = []
        self.num = []

    def info(self):
        """
        Display dataset information.

        Returns
        -------
        None
            Prints the output of DataFrame.info().
        """
        return self.data.info()
        
    def describe(self):
        """
        Generate descriptive statistics for numerical columns.

        Returns
        -------
        pandas.DataFrame
            A transposed summary statistics table.
        """
        return self.data.describe().T
    
    def nullValues(self):
        """
        Count missing values in each column.

        Returns
        -------
        pandas.Series
            A series showing the number of null values per column.
        """
        return self.data.isnull().sum()
    
    def dtypes(self):
        """
        Return the data types of all columns.

        Returns
        -------
        pandas.Series
            A series mapping column names to their data types.
        """
        return self.data.dtypes
    
    def giveTarget(self): 
        """
        Return the name of the target column.

        Returns
        -------
        str
            The target column name.
        """                      
        return self.target


    def setCol(self):
        self.col = self.data.columns
        return self.col
        
    def setCatList(self, catList=None):
        """
        Set the list of categorical column names.

        If no list is provided, the method uses the columns detected in `cat_df`.
        Otherwise, it uses the user-provided list.

        Parameters
        ----------
        catList : list, optional
            A list of column names to treat as categorical.

        Returns
        -------
        list
            The updated list of categorical columns.
        """
        if catList is None:                                                 
            self.catList = self.data.select_dtypes(include=['object', 'category']).columns.tolist()     # If user has not specified a list, create a list of all categorical columns in cat_df
        else:
            self.catList=catList                            # Else use user-provided list
        return self.catList
    
    def setNumList(self, numList=None):
        """
        Set the list of numerical column names.

        If no list is provided, the method uses the numeric columns detected
        in `num_df`. The target column is automatically removed if present.

        Parameters
        ----------
        numList : list, optional
            A list of column names to treat as numeric.

        Returns
        -------
        list
            The updated list of numeric columns.
        """

        if numList is None:                                             
            self.numList = self.data.select_dtypes(include=['number']).columns.tolist()                 # If user has not specified a list, create a list of all numeric columns in num_df
        
        else:
            self.numList=numList                                        # Else use user-provided list
        
        return self.numList
    
    def remove_target_from_features(self):
        """
        Ensure the target column is not included in any feature list.

        This method removes the target column from both `numList` and `catList`
        if it appears in either. It is intended to be called after feature lists
        have been constructed (e.g., after running setNumList, setCatList, or
        detectBoolean) to guarantee that the target is never treated as an input
        feature during model training.

        Returns
        -------
        None
        """
        if self.target in self.numList:
            self.numList.remove(self.target)
        if self.target in self.catList:
            self.catList.remove(self.target)

    #-------------------------------------------------------------------------------------------------------------------------------------    
    # EXPLORE CATEGORICAL FEATURES

    def setCatdf(self):
        """
        Create a DataFrame containing only categorical columns.

        Returns
        -------
        pandas.DataFrame
            A DataFrame of object or category dtype columns.
        """

        self.cat_df = self.data.select_dtypes(include=['object', 'category'])
        return self.cat_df
    
    def countPlots(self, splitTarg=False, show=True):
        """
        Generate count plots for all categorical columns.

        Parameters
        ----------
        splitTarg : bool, default False
            If True, the target column is used as the hue variable.
        show : bool, default True
            If True, displays the plot. If False, returns the figure object.

        Returns
        -------
        matplotlib.figure.Figure or None
            The figure object if show=False, otherwise None.
        """

        if len(self.catList) == 0:
            print('No categorical columns to plot')
            return
        else:
            n = len(self.catList)
            cols = 2
            figure, ax = plt.subplots(math.ceil(n/cols), cols)
            r = 0
            c = 0
            for col in self.catList:
                if splitTarg == False:
                    sns.countplot(data=self.data, x=col, ax=ax[r][c])
                if splitTarg == True:
                    sns.countplot(data=self.data, x=col, hue=self.target, ax=ax[r][c])
                c += 1
                if c == cols:
                    r += 1
                    c = 0
            figure.suptitle('Distribution of Categorical Columns', fontsize=16, y=1.02)
            plt.tight_layout()


    #-------------------------------------------------------------------------------------------------------------------------------------
    # EXPLORE NUMERIC FEATURES

    def setNumdf(self):
        """
        Create a DataFrame containing only numeric columns.

        Returns
        -------
        pandas.DataFrame
            A DataFrame of numeric columns.
        """
        self.num_df = self.data.select_dtypes(include='number')         # Create df containing only numeric columns

    def corrMatrix(self):
        """
        Compute the correlation of all numeric features with the target.

        Returns
        -------
        pandas.Series or None
            A sorted series of correlations with the target column.
            Returns None if the target is not numeric.
        """

        if self.target not in self.num_df.columns:                                  # If the target is not in num_df, continue without creating matrix
            print("Target must be numeric to compute correlations")
            return None
        else:                                                                       # Else calc num_df correlation matrix
            return self.num_df.corr()[self.target].sort_values(ascending=False)         
   
    def detectDist(self, fitColumns=None, method='parametric', verbose=0):
        """
        Fit probability distributions to numeric columns using distfit.

        Parameters
        ----------
        fitColumns : list, optional
            Columns to fit. Defaults to `numList`.
        method : str, default 'parametric'
            The fitting method used by distfit.
        verbose : int, default 0
            Verbosity level for distfit.

        Returns
        -------
        dict
            A dictionary mapping column names to best-fit distribution results.

        Documentation
        -------
            distfit: https://erdogant.github.io/distfit/pages/html/index.html
        """

        if fitColumns is None:                  # fitColumns=None gives user option to change list
            fitColumns = self.numList           # If user does not specify columns, default is numList

        self.dFit_results = {}
        self.dFit_models = {}
        param_labels = ['DOF', 'Location', 'scale']

        # Fit the data
        for column in fitColumns:
            dfit = distfit(method=method, verbose=verbose)      # Initialize
            dfit.fit_transform(self.data[column])               # Fit and score data

            best_dist = dfit.model['name']                              # Name of best model
            best_params = dict(zip(param_labels, dfit.model['params']))  # Parameters of best model
        
            self.dFit_results[column] = {
                'Best-fit Model': best_dist,
                'Model Parameters': best_params
            }
            self.dFit_models[column] = dfit
        return self.dFit_results

    def printDistResults(self):
        """
        Print the best-fit distribution results for each numeric column.

        Returns
        -------
        None
            Prints formatted distribution information.
        """

        print("DISTRIBUTION TYPES:\n")
        for col, info in self.dFit_results.items():
            print(f"{col}:")
            print(f"  Best-fit Model: {info['Best-fit Model']}")
            print("  Parameters:")
            for p, v in info['Model Parameters'].items():
                print(f"    {p}: {float(v):.4f}")
            print()  # Blank line between columns

    def detectOutliers_IQR(self, col):
        """
        Detect outliers in a numeric column using the IQR method.

        Parameters
        ----------
        col : str
            The column name to analyze.

        Returns
        -------
        tuple
            (lower_bound, upper_bound, count_below, count_above)
        """

        Q1 = self.data[col].quantile(0.25)
        Q3 = self.data[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        lower_mask = self.data[col] < lower
        upper_mask = self.data[col] > upper

        count_low = lower_mask.sum()
        count_high = upper_mask.sum()

        return lower, upper, count_low, count_high

    def histPlots(self, kde=True, splitTarg=False, show=True):
        """
        Generate histogram plots for all numeric columns.

        Includes:
        - optional KDE curves,
        - optional target-based splitting,
        - best-fit distribution annotations (if available),
        - IQR outlier boundaries.

        Parameters
        ----------
        kde : bool, default True
            Whether to overlay a KDE curve.
        splitTarg : bool, default False
            Whether to split histograms by the target column.
        show : bool, default True
            If True, displays the plot. If False, returns the figure.

        Returns
        -------
        matplotlib.figure.Figure or None
            The figure object if show=False, otherwise None.
        """

        n = len(self.numList)
        cols = 2
        rows = math.ceil(n/cols)

        figure, ax = plt.subplots(rows, cols, figsize=(12, 4*rows))
        r = 0
        c = 0

        for col in self.numList:
            #print("r:",r,"c:",c)
            if splitTarg == False:
                sns.histplot(data=self.data, x=col, kde=kde, ax=ax[r][c])
            if splitTarg == True:
                sns.histplot(data=self.data, x=col, hue=self.target, kde=kde, ax=ax[r][c])
            
            dist_info = self.dFit_results.get(col, None)

            if dist_info is not None:
                best_model = dist_info['Best-fit Model']
                params = dist_info['Model Parameters']

                # Format parameters nicely
                param_text = "\n".join([f"{k}: {v:.3f}" for k, v in params.items()])

                # Add text annotation to subplot
                ax[r][c].text(
                    0.98, 0.98,
                    f"Best-fit: {best_model}\n{param_text}",
                    transform=ax[r][c].transAxes,
                    ha='right', va='top',
                    fontsize=9,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
                )

            lower, upper, n_low, n_high = self.detectOutliers_IQR(col)

            ax[r][c].axvline(lower, color='red', linestyle='--', linewidth=1, label=f'Lower IQR: {lower}')
            ax[r][c].axvline(upper, color='red', linestyle='--', linewidth=1, label=f'Upper IQR: {upper}')
            ax[r][c].legend()

            c += 1
            if c == cols:
                r += 1
                c = 0
            
            figure.suptitle('Distribution of Numeric Columns', fontsize=16, y=1.02)
            plt.tight_layout()

    #-------------------------------------------------------------------------------------------------------------------------------------
    def fullEDA(self):
        out1 = widgets.Output()
        out2 = widgets.Output()
        out3 = widgets.Output()
        out4 = widgets.Output()

        tab = widgets.Tab(children = [out1, out2, out3])
        tab.set_title(0, 'Info')
        tab.set_title(1, 'Categorical')
        tab.set_title(2, 'Numerical')
        display(tab)

        with out1:
            self.info()

        with out2:
            fig2 = self.countPlots(splitTarg=True, show=False)
            plt.show(fig2)
        
        with out3:
            fig3 = self.histPlots(kde=True, show=False)
            plt.show(fig3)

#-------------------------------------------------------------------------------------------------------------------------------------

    def run_basicEDA(self):
        """
        Run a basic exploratory data analysis workflow.

        This method prints a high-level overview of the dataset, including:
        - DataFrame info (column types, memory usage)
        - Summary statistics for numeric columns
        - Null value counts
        - Automatic detection of categorical and numeric columns
        - Boolean-like column detection
        - Data types summary
        - Target column name

        The method updates internal attributes such as:
        - catList
        - numList
        - cat_df
        - num_df

        Returns
        -------
        None
            Outputs are printed to the console.
        """

        print('INFO:')
        print(self.info())

        print('\nBASIC STATS:')
        print(self.describe())

        print('\nNULL VALUES:')
        print(self.nullValues())

        #print('\nCOLUMNS WITH DTYPES CORRECTIONS:')
        #print(self.fixNumericTypes())

        self.setCatList()
        self.setNumList()
        self.remove_target_from_features()
        #self.detectBoolean()

        self.setCatdf()
        self.setNumdf()

        print('\nDATA TYPES:')
        print(self.dtypes())

        print('\nTARGET:')        
        print(self.giveTarget())

    def run_catEDA(self):
        """
        Run exploratory analysis for categorical features.

        This method generates count plots for all categorical columns
        currently stored in `catList`. If no categorical columns exist,
        the underlying plotting function will notify the user.

        Returns
        -------
        None
            Displays categorical distribution plots.
        """
        self.countPlots()

    def run_numEDA(self):
        """
        Run exploratory analysis for numeric features.

        This method performs:
        - Correlation analysis between numeric features and the target
        - Distribution fitting using distfit
        - Histogram plotting with KDE, best-fit distribution annotations,
        and IQR-based outlier boundaries

        Returns
        -------
        None
            Prints correlation results and displays numeric distribution plots.
        """
        print('\nCORRELATION MATRIX:')
        print(self.corrMatrix())
        
        self.detectDist()

        self.histPlots()


#-------------------------------------------------------------------------------------------------------------------------------------
    # Run all EDA functios
    def run_allEDA(self):
        """
        Run a full exploratory data analysis workflow.

        This method combines the functionality of basic, categorical,
        and numeric EDA routines. It performs:

        - Dataset info summary
        - Descriptive statistics
        - Null value counts
        - Automatic detection of categorical, numeric, and Boolean-like columns
        - Creation of cat_df and num_df
        - Data type summary
        - Target column display
        - Count plots for categorical features
        - Correlation matrix for numeric features
        - Distribution fitting for numeric columns
        - Histogram plots with KDE, distribution annotations, and outlier markers

        Returns
        -------
        None
            Prints summaries and displays all EDA visualizations.
        """
        
        print('INFO:')
        print(self.info())

        print('\nBASIC STATS:')
        print(self.describe())

        print('\nNULL VALUES:')
        print(self.nullValues())

        #print('\nCOLUMNS WITH DTYPES CORRECTIONS:')
        #print(self.fixNumericTypes())

        self.setCatList()
        self.setNumList()
        self.remove_target_from_features()
        #self.detectBoolean()

        self.setCatdf()
        self.setNumdf()

        print('\nDATA TYPES:')
        print(self.dtypes())

        print('\nTARGET:')        
        print(self.giveTarget())

    # CATEGORICAL COLUMNS:
        self.countPlots()

    # NUMERIC COLUMNS:
        print('\nCORRELATION MATRIX:')
        print(self.corrMatrix())
        
        self.detectDist()

        self.histPlots()
