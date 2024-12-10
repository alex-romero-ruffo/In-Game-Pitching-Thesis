import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import plotly.express as px
import plotly.graph_objs as go
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QComboBox, QTabWidget, QListWidget,
    QListWidgetItem, QAbstractItemView, QSplitter, QMessageBox, QFileDialog, QSizePolicy
)
from PyQt5.QtCore import Qt
from analysis import load_data, get_unique_values, filter_data, compute_correlation_matrix

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(message)s')

class BaseballAnalyticsApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Baseball Pitch Analytics Tool")
        self.setGeometry(100, 100, 1600, 900)
        self.data = None
        self.filtered_data = None
        self.updating_filters = False
        self.initUI()

    def initUI(self):
        self.loadButton = QPushButton('Load Data')
        self.loadButton.clicked.connect(self.loadData)

        # Tabs
        self.tabs = QTabWidget()
        self.regression_tab = QWidget()
        self.correlation_tab = QWidget()
        self.time_series_tab = QWidget()

        self.tabs.addTab(self.regression_tab, "Regression Analysis")
        self.tabs.addTab(self.correlation_tab, "Correlation Matrix")
        self.tabs.addTab(self.time_series_tab, "Time-Series Analysis")

        # Filters
        self.dateList = QListWidget()
        self.dateList.setSelectionMode(QAbstractItemView.MultiSelection)
        self.dateList.itemSelectionChanged.connect(self.onFilterChanged)

        self.gameList = QListWidget()
        self.gameList.setSelectionMode(QAbstractItemView.MultiSelection)
        self.gameList.itemSelectionChanged.connect(self.onFilterChanged)

        self.pitcherList = QListWidget()
        self.pitcherList.setSelectionMode(QAbstractItemView.MultiSelection)
        self.pitcherList.itemSelectionChanged.connect(self.onFilterChanged)

        self.pitchTypeList = QListWidget()
        self.pitchTypeList.setSelectionMode(QAbstractItemView.MultiSelection)
        self.pitchTypeList.itemSelectionChanged.connect(self.onFilterChanged)

        self.selectedFiltersLabel = QLabel("Selected Filters:")
        self.selectedFiltersLabel.setWordWrap(True)
        self.selectedFiltersLabel.setMaximumHeight(100)

        filterLayout = QGridLayout()
        filterLayout.addWidget(QLabel('Filter by Date:'), 0, 0)
        filterLayout.addWidget(self.dateList, 1, 0)
        filterLayout.addWidget(QLabel('Filter by Game:'), 0, 1)
        filterLayout.addWidget(self.gameList, 1, 1)
        filterLayout.addWidget(QLabel('Filter by Pitcher:'), 0, 2)
        filterLayout.addWidget(self.pitcherList, 1, 2)
        filterLayout.addWidget(QLabel('Filter by Pitch Type:'), 0, 3)
        filterLayout.addWidget(self.pitchTypeList, 1, 3)
        filterLayout.addWidget(self.selectedFiltersLabel, 2, 0, 1, 4)

        for list_widget in [self.dateList, self.gameList, self.pitcherList, self.pitchTypeList]:
            list_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            list_widget.setMinimumSize(200, 300) 
        filterWidget = QWidget()
        filterWidget.setLayout(filterLayout)

        # Regression Tab Layout
        regLayout = QVBoxLayout()
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('X Variable:'))
        self.xComboBox = QComboBox()  # Removed walrus operator
        hbox.addWidget(self.xComboBox)
        hbox.addWidget(QLabel('Y Variable:'))
        self.yComboBox = QComboBox()  # Removed walrus operator
        hbox.addWidget(self.yComboBox)
        regLayout.addLayout(hbox)
        self.analyzeButton = QPushButton('Analyze Regression')  # Removed walrus operator
        self.analyzeButton.clicked.connect(self.analyzeRegression)
        regLayout.addWidget(self.analyzeButton)
        self.analyzeButton.setEnabled(False)
        self.regression_tab.setLayout(regLayout)

        # Correlation Tab Layout
        corrLayout = QVBoxLayout()
        self.corrButton = QPushButton('Show Correlation Matrix')  # Removed walrus operator
        self.corrButton.clicked.connect(self.showCorrelationMatrix)
        self.corrButton.setEnabled(False)
        corrLayout.addWidget(self.corrButton)
        self.correlation_tab.setLayout(corrLayout)

        # Time-Series Tab Layout
        timeLayout = QVBoxLayout()
        hbox_time = QHBoxLayout()
        hbox_time.addWidget(QLabel('Time Variable:'))
        self.timeVariableComboBox = QComboBox()  # Removed walrus operator
        self.timeVariableComboBox.addItems(['game_date', 'at_bat_number', 'pitch_number'])
        hbox_time.addWidget(self.timeVariableComboBox)
        hbox_time.addWidget(QLabel('Y Variable:'))
        self.timeYVariableComboBox = QComboBox()  # Removed walrus operator
        hbox_time.addWidget(self.timeYVariableComboBox)
        timeLayout.addLayout(hbox_time)
        self.timeAnalyzeButton = QPushButton('Analyze Time Series')  # Removed walrus operator
        self.timeAnalyzeButton.clicked.connect(self.analyzeTimeSeries)
        timeLayout.addWidget(self.timeAnalyzeButton)
        self.timeAnalyzeButton.setEnabled(False)
        self.time_series_tab.setLayout(timeLayout)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(filterWidget)
        splitter.addWidget(self.tabs)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)  

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.loadButton)
        mainLayout.addWidget(splitter)

        container = QWidget()
        container.setLayout(mainLayout)
        self.setCentralWidget(container)


    def loadData(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Data File", "", "CSV Files (*.csv);;All Files (*)", options=options)
        if not file_path:
            return

        try:
            self.data = load_data(file_path)
        except Exception as e:
            logging.error(f"Error in loadData: {e}")
            QMessageBox.critical(self, "Error", f"Failed to load data: {e}")
            return

        self.filtered_data = self.data.copy()
        self.populateFilters()
        self.corrButton.setEnabled(True)
        self.timeAnalyzeButton.setEnabled(True)
        self.analyzeButton.setEnabled(True)

        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        self.xComboBox.clear()
        self.yComboBox.clear()
        self.xComboBox.addItems(numeric_cols)
        self.yComboBox.addItems(numeric_cols)

        self.timeYVariableComboBox.clear()
        self.timeYVariableComboBox.addItems(numeric_cols)

    def populateFilters(self):
        self.updating_filters = True
        dates = self.data['game_date'].drop_duplicates().sort_values()
        self.dateList.blockSignals(True)
        self.dateList.clear()
        for date in dates:
            date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
            item = QListWidgetItem(date_str)
            item.setData(Qt.UserRole, date)
            self.dateList.addItem(item)
        self.dateList.blockSignals(False)
        # Do not select all dates by default

        # Initially populate the game, pitcher, and pitch type lists
        self.updateGamesFromDate()
        self.updatePitchers()
        self.updatePitchTypes()
        self.updating_filters = False
        self.applyFilters()

    def onFilterChanged(self):
        if not self.updating_filters:
            self.updating_filters = True
            sender = self.sender()
            if sender == self.dateList:
                self.updateGamesFromDate()
                self.updatePitchers()
                self.updatePitchTypes()
            elif sender == self.gameList:
                self.updatePitchers()
                self.updatePitchTypes()
            elif sender == self.pitcherList:
                self.updatePitchTypes()
            elif sender == self.pitchTypeList:
                self.updatePitchers()
            self.updating_filters = False
            self.applyFilters()

    def updateGamesFromDate(self):
        logging.debug("Updating games based on selected dates...")
        selected_dates = [item.data(Qt.UserRole) for item in self.dateList.selectedItems()]
        logging.debug(f"Selected dates: {selected_dates}")

        self.gameList.blockSignals(True)
        self.gameList.clear()
        if selected_dates:
            games_on_selected_dates = self.data[self.data['game_date'].isin(selected_dates)][['game_pk', 'game_date']].drop_duplicates()
            for idx, row in games_on_selected_dates.iterrows():
                game_pk = row['game_pk']
                game_date = row['game_date'].strftime('%Y-%m-%d') if isinstance(row['game_date'], pd.Timestamp) else str(row['game_date'])
                item_text = f"Game {game_pk} on {game_date}"
                item = QListWidgetItem(item_text)
                item.setData(Qt.UserRole, game_pk)
                self.gameList.addItem(item)
        self.gameList.blockSignals(False)

    def updatePitchers(self):
        logging.debug("Updating pitchers based on selected games and pitch types...")
        selected_games = [item.data(Qt.UserRole) for item in self.gameList.selectedItems()]
        selected_pitch_types = [item.text() for item in self.pitchTypeList.selectedItems()]
        logging.debug(f"Selected games: {selected_games}")
        logging.debug(f"Selected pitch types: {selected_pitch_types}")
        df = self.data.copy()
        if selected_games:
            df = df[df['game_pk'].isin(selected_games)]
        if selected_pitch_types:
            df = df[df['pitch_type'].isin(selected_pitch_types)]

        pitchers = get_unique_values(df, 'player_name')
        logging.debug(f"Available pitchers: {pitchers}")

        selected_pitchers = [item.text() for item in self.pitcherList.selectedItems()]
        self.pitcherList.blockSignals(True)
        self.pitcherList.clear()
        for pitcher in pitchers:
            item = QListWidgetItem(pitcher)
            self.pitcherList.addItem(item)
            if pitcher in selected_pitchers:
                item.setSelected(True)
        self.pitcherList.blockSignals(False)


    def updatePitchTypes(self):
        logging.debug("Updating pitch types based on selected games and pitchers...")
        selected_games = [item.data(Qt.UserRole) for item in self.gameList.selectedItems()]
        selected_pitchers = [item.text() for item in self.pitcherList.selectedItems()]
        logging.debug(f"Selected games: {selected_games}")
        logging.debug(f"Selected pitchers: {selected_pitchers}")
        df = self.data.copy()
        if selected_games:
            df = df[df['game_pk'].isin(selected_games)]
        if selected_pitchers:
            df = df[df['player_name'].isin(selected_pitchers)]

        pitch_types = get_unique_values(df, 'pitch_type')
        logging.debug(f"Available pitch types: {pitch_types}")

        selected_pitch_types = [item.text() for item in self.pitchTypeList.selectedItems()]
        self.pitchTypeList.blockSignals(True)
        self.pitchTypeList.clear()
        for pitch in pitch_types:
            item = QListWidgetItem(pitch)
            self.pitchTypeList.addItem(item)
            if pitch in selected_pitch_types:
                item.setSelected(True)
        self.pitchTypeList.blockSignals(False)


    def applyFilters(self):
        if self.updating_filters:
            return
        logging.debug("Applying filters...")
        selected_dates = [item.data(Qt.UserRole) for item in self.dateList.selectedItems()]
        selected_games = [item.data(Qt.UserRole) for item in self.gameList.selectedItems()]
        selected_pitchers = [item.text() for item in self.pitcherList.selectedItems()]
        selected_pitch_types = [item.text() for item in self.pitchTypeList.selectedItems()]

        logging.debug(f"Filters - Dates: {selected_dates}, Games: {selected_games}, Pitchers: {selected_pitchers}, Pitch Types: {selected_pitch_types}")

        df = self.data.copy()
        df.reset_index(drop=True, inplace=True)

        if selected_dates:
            df = df[df['game_date'].isin(selected_dates)]
            df.reset_index(drop=True, inplace=True)
        if selected_games:
            df = df[df['game_pk'].isin(selected_games)]
            df.reset_index(drop=True, inplace=True)
        if selected_pitchers:
            df = df[df['player_name'].isin(selected_pitchers)]
            df.reset_index(drop=True, inplace=True)
        if selected_pitch_types:
            df = df[df['pitch_type'].isin(selected_pitch_types)]
            df.reset_index(drop=True, inplace=True)

        self.filtered_data = df
        self.updateSelectedFiltersLabel()


    def updateSelectedFiltersLabel(self):
        def format_selected_items(items):
            if len(items) > 5:
                return ', '.join(items[:5]) + f", ... ({len(items)} total)"
            elif len(items) == 0:
                return "None"
            else:
                return ', '.join(items)

        selected_dates = [item.text() for item in self.dateList.selectedItems()]
        selected_games = [item.text() for item in self.gameList.selectedItems()]
        selected_pitchers = [item.text() for item in self.pitcherList.selectedItems()]
        selected_pitch_types = [item.text() for item in self.pitchTypeList.selectedItems()]

        filters_text = f"Dates: {format_selected_items(selected_dates)}\n"
        filters_text += f"Games: {format_selected_items(selected_games)}\n"
        filters_text += f"Pitchers: {format_selected_items(selected_pitchers)}\n"
        filters_text += f"Pitch Types: {format_selected_items(selected_pitch_types)}"

        self.selectedFiltersLabel.setText(f"Selected Filters:\n{filters_text}")

    def analyzeRegression(self):
        x_var = self.xComboBox.currentText()
        y_var = self.yComboBox.currentText()
        if x_var and y_var:
            try:
                df = self.filtered_data[[x_var, y_var]].dropna()
                if df.empty:
                    QMessageBox.warning(self, "No Data", "No data available after filtering and dropping NaNs.")
                    return
                # Calculate correlation coefficient
                corr_coef = df.corr().iloc[0, 1]
                # Create regression plot
                plt.figure(figsize=(8, 6))
                sns.regplot(x=df[x_var], y=df[y_var], line_kws={'color':'red'})
                plt.xlabel(x_var)
                plt.ylabel(y_var)
                plt.title(f'Regression between {x_var} and {y_var}\nCorrelation Coefficient: {corr_coef:.2f}')
                plt.tight_layout()
                plt.show()
            except Exception as e:
                logging.error(f"Error in analyzeRegression: {e}")
                QMessageBox.critical(self, "Error", f"An error occurred during regression analysis:\n{e}")

    def showCorrelationMatrix(self, show_rows=True):
        try:
            if self.filtered_data is not None and not self.filtered_data.empty:
                if show_rows:
                    print("Rows used in the correlation matrix:")
                    print(self.filtered_data.dropna())  # Dropping NaNs to show clean rows

                # Compute the correlation matrix
                corr_matrix = compute_correlation_matrix(self.filtered_data)
                if corr_matrix.empty:
                    QMessageBox.warning(self, "No Data", "No numerical data available for correlation matrix.")
                    return

                hover_text = [[f'{x} vs {y}<br>Correlation: {corr_matrix.loc[x, y]:.2f}' 
                            for y in corr_matrix.columns] 
                            for x in corr_matrix.index]

                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values,
                    x=corr_matrix.columns,
                    y=corr_matrix.index,
                    text=hover_text, 
                    hoverinfo='text',
                    colorscale='RdBu',
                    zmin=-1, zmax=1, 
                    colorbar=dict(title="Correlation")
                ))

                fig.update_layout(
                    title='Correlation Matrix Heatmap',
                    xaxis_nticks=len(corr_matrix.columns),
                    yaxis_nticks=len(corr_matrix.index),
                    xaxis_tickangle=45, 
                    height=800, 
                    margin=dict(l=100, r=100, b=150, t=100),  
                )

                fig.show()

                print("\nCorrelation Matrix Values:\n")
                print(corr_matrix)

            else:
                QMessageBox.warning(self, "No Data", "No data available to compute correlation matrix.")
        except Exception as e:
            logging.error(f"Error in showCorrelationMatrix: {e}")
            QMessageBox.critical(self, "Error", f"An error occurred while displaying the correlation matrix:\n{e}")


    def analyzeTimeSeries(self):
        time_var = self.timeVariableComboBox.currentText()
        y_var = self.timeYVariableComboBox.currentText()
        if time_var and y_var:
            try:
                df = self.filtered_data[[time_var, y_var]].dropna()
                if df.empty:
                    QMessageBox.warning(self, "No Data", "No data available after filtering and dropping NaNs.")
                    return
                if time_var == 'game_date':
                    df.sort_values(by=time_var, inplace=True)
                plt.figure(figsize=(10, 6))
                sns.lineplot(x=df[time_var], y=df[y_var])
                plt.xlabel(time_var)
                plt.ylabel(y_var)
                plt.title(f'{y_var} over {time_var}')
                plt.tight_layout()
                plt.show()
            except Exception as e:
                logging.error(f"Error in analyzeTimeSeries: {e}")
                QMessageBox.critical(self, "Error", f"An error occurred during time series analysis:\n{e}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BaseballAnalyticsApp()
    window.show()
    sys.exit(app.exec_())
