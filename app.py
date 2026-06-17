"""
AI Classifier Lab - Iris Flower Classification
Group: GroupAlpha
A production-ready Streamlit application for interactive ML classification.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from datetime import datetime
import json
import os

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize Streamlit session state variables."""
    if "run_history" not in st.session_state:
        st.session_state.run_history = []
    if "current_result" not in st.session_state:
        st.session_state.current_result = None
    if "data" not in st.session_state:
        st.session_state.data = None
    if "preprocessed" not in st.session_state:
        st.session_state.preprocessed = None


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

def load_data(path):
    """
    Load dataset from a CSV file path.

    Parameters:
        path (str): Absolute or relative path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset, or None if file not found.
    """
    try:
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        st.error(f"Dataset file not found at: `{path}`. Please check the path and try again.")
        return None
    except pd.errors.EmptyDataError:
        st.error("The dataset file is empty. Please provide a valid CSV file.")
        return None
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None


def preprocess_data(data):
    """
    Preprocess the dataset: encode target labels, split features/target,
    and scale feature values.

    Parameters:
        data (pd.DataFrame): Raw dataset with features and target column.

    Returns:
        dict: Preprocessed data containing:
            - X_train, X_test, y_train, y_test
            - feature_names, target_name, classes
            - scaler, label_encoder
    """
    if data is None or data.empty:
        st.error("No data provided for preprocessing.")
        return None

    # Identify target column (last column assumed to be target)
    target_col = data.columns[-1]
    feature_cols = list(data.columns[:-1])

    # Separate features and target
    X = data[feature_cols].copy()
    y = data[target_col].copy()

    # Validate that we have enough data
    if len(X) < 10:
        st.error("Dataset too small. Need at least 10 samples.")
        return None

    # Handle missing values
    if X.isnull().sum().sum() > 0:
        X = X.fillna(X.mean(numeric_only=True))

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y.astype(str))

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    return {
        "X": X_scaled,
        "y": y_encoded,
        "feature_names": feature_cols,
        "target_name": target_col,
        "classes": list(le.classes_),
        "scaler": scaler,
        "label_encoder": le,
    }


# =============================================================================
# CORE ML LOGIC
# =============================================================================

def run_model_or_algorithm(data, params):
    """
    Train and evaluate a Scikit-learn classifier based on user parameters.

    Parameters:
        data (dict): Preprocessed data from preprocess_data().
        params (dict): User-selected parameters including:
            - algorithm: "Logistic Regression" or "Random Forest"
            - test_size: float between 0.1 and 0.5
            - random_state: int for reproducibility
            - max_depth: int or None (for Random Forest)
            - n_estimators: int (for Random Forest)
            - C: float regularization strength (for Logistic Regression)

    Returns:
        dict: Training result containing model, metrics, predictions, and params.
    """
    if data is None:
        return None

    # Extract data
    X = data["X"]
    y = data["y"]
    feature_names = data["feature_names"]
    classes = data["classes"]

    # Split data
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError as e:
        st.error(f"Train/test split failed: {e}")
        return None

    # Initialize model based on algorithm selection
    algorithm = params.get("algorithm", "Logistic Regression")

    if algorithm == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=params.get("n_estimators", 100),
            max_depth=params.get("max_depth", None),
            random_state=random_state,
        )
    else:
        model = LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=1000,
            random_state=random_state,
        )

    # Train model
    with st.spinner(f"Training {algorithm}..."):
        model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    # Handle multiclass averaging for precision and recall
    avg_method = "weighted" if len(classes) > 2 else "binary"
    precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
    recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=classes, output_dict=True)

    # Feature importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0]) if model.coef_.ndim > 1 else np.abs(model.coef_)
    else:
        importances = np.zeros(len(feature_names))

    feature_importance = dict(zip(feature_names, importances))

    return {
        "model": model,
        "algorithm": algorithm,
        "params": params,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_importance": feature_importance,
        "classes": classes,
        "feature_names": feature_names,
        "n_samples": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


# =============================================================================
# EXPLAINABILITY
# =============================================================================

def generate_explanation(result, context=None):
    """
    Generate a natural-language explanation of model predictions and
    display feature importances or decision rules.

    Parameters:
        result (dict): Output from run_model_or_algorithm().
        context (str): Optional additional context string.

    Returns:
        str: Natural language explanation of the model results.
    """
    if result is None:
        return "No results available to explain."

    algorithm = result["algorithm"]
    accuracy = result["accuracy"]
    precision = result["precision"]
    recall = result["recall"]
    classes = result["classes"]
    feature_importance = result["feature_importance"]
    params = result["params"]

    # Determine top features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_feature = sorted_features[0][0]
    top_value = sorted_features[0][1]

    # Build explanation
    explanation = f"""
    ## Model Explanation

    This experiment used **{algorithm}** to classify Iris flowers into three species:
    **{', '.join(classes)}**. The model was trained on **{result['n_train']} samples**
    and evaluated on **{result['n_test']} samples** with a test-size split of
    **{params.get('test_size', 0.2) * 100:.0f}%**.

    ### Key Performance Insights

    - **Accuracy: {accuracy:.2%}** — The model correctly predicted the species
      for {accuracy:.2%} of the test flowers.
    - **Precision: {precision:.2%}** — When the model predicts a class, it is
      correct {precision:.2%} of the time on average.
    - **Recall: {recall:.2%}** — The model successfully identifies {recall:.2%}
      of the actual instances for each class on average.

    ### Most Influential Feature

    The feature with the highest predictive power is **{top_feature}**
    (importance score: {top_value:.4f}). This means that {top_feature}
    contributes the most to the model's decision-making process when
    distinguishing between the three Iris species.

    ### Hyperparameter Summary
    """

    if algorithm == "Random Forest":
        explanation += f"""
    - Number of estimators (trees): **{params.get('n_estimators', 100)}**
    - Maximum tree depth: **{params.get('max_depth', 'None (unlimited)')}**
    - Random state: **{params.get('random_state', 42)}** (for reproducibility)
        """
    else:
        explanation += f"""
    - Regularization strength (C): **{params.get('C', 1.0)}**
    - Random state: **{params.get('random_state', 42)}** (for reproducibility)
        """

    if context:
        explanation += f"\n\n**Additional Context:** {context}"

    # Add recommendation based on accuracy
    if accuracy >= 0.95:
        explanation += """

    **Verdict:** This is an **excellent** model performance. The classifier
    achieves near-perfect accuracy on this dataset, which is expected for
    the Iris dataset since the classes are well-separated in feature space.
        """
    elif accuracy >= 0.85:
        explanation += """

    **Verdict:** This is a **good** model performance. The classifier reliably
    distinguishes between species with high accuracy.
        """
    else:
        explanation += """

    **Verdict:** This performance is **below expectations** for the Iris dataset.
    Consider tuning hyperparameters or trying a different algorithm.
        """

    return explanation


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_visuals(data, result):
    """
    Generate Plotly interactive charts including feature distributions,
    scatter plots, and a confusion matrix heatmap.

    Parameters:
        data (dict): Preprocessed data from preprocess_data().
        result (dict): Output from run_model_or_algorithm().

    Returns:
        dict: Dictionary of Plotly figure objects.
    """
    if data is None or result is None:
        return {}

    figures = {}

    # --- 1. Confusion Matrix Heatmap ---
    cm = result["confusion_matrix"]
    classes = result["classes"]

    cm_fig = ff.create_annotated_heatmap(
        z=cm,
        x=classes,
        y=classes,
        colorscale="Blues",
        showscale=True,
        annotation_text=cm.astype(str),
    )
    cm_fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label",
        height=450,
    )
    figures["confusion_matrix"] = cm_fig

    # --- 2. Feature Importance Bar Chart ---
    feature_importance = result["feature_importance"]
    feat_df = pd.DataFrame({
        "Feature": list(feature_importance.keys()),
        "Importance": list(feature_importance.values()),
    }).sort_values("Importance", ascending=True)

    feat_fig = px.bar(
        feat_df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Teal",
        title="Feature Importance",
    )
    feat_fig.update_layout(height=400, yaxis_title="", xaxis_title="Importance Score")
    figures["feature_importance"] = feat_fig

    # --- 3. Metrics Gauge Chart ---
    metrics_fig = go.Figure()

    metrics = [
        ("Accuracy", result["accuracy"], "#2E8B57"),
        ("Precision", result["precision"], "#4682B4"),
        ("Recall", result["recall"], "#DAA520"),
    ]

    for i, (name, value, color) in enumerate(metrics):
        metrics_fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={"x": [i / 3, (i + 1) / 3 - 0.02], "y": [0, 1]},
            title={"text": name, "font": {"size": 14}},
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 1], "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.75},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "#cccccc",
                "steps": [
                    {"range": [0, 0.5], "color": "#ffcccc"},
                    {"range": [0.5, 0.8], "color": "#ffffcc"},
                    {"range": [0.8, 1], "color": "#ccffcc"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 2},
                    "thickness": 0.8,
                    "value": 0.9,
                },
            },
        ))

    metrics_fig.update_layout(
        title="Performance Metrics Overview",
        height=350,
        margin=dict(l=30, r=30, t=80, b=20),
    )
    figures["metrics_gauge"] = metrics_fig

    # --- 4. Scatter Plot (first two features colored by true labels) ---
    X_test = result["X_test"]
    y_test = result["y_test"]
    feature_names = result["feature_names"]
    le_classes = result["classes"]

    scatter_df = pd.DataFrame(X_test, columns=feature_names)
    scatter_df["True Label"] = [le_classes[y] for y in y_test]
    scatter_df["Predicted Label"] = [le_classes[y] for y in result["y_pred"]]
    scatter_df["Correct"] = scatter_df["True Label"] == scatter_df["Predicted Label"]

    if len(feature_names) >= 2:
        scatter_fig = px.scatter(
            scatter_df,
            x=feature_names[0],
            y=feature_names[1],
            color="True Label",
            symbol="Correct",
            size_max=12,
            opacity=0.8,
            title=f"Test Set Predictions: {feature_names[0]} vs {feature_names[1]}",
            color_discrete_sequence=px.colors.qualitative.Set1,
        )
        scatter_fig.update_traces(marker=dict(size=12))
        scatter_fig.update_layout(height=500, legend_title_text="Species")
        figures["scatter_plot"] = scatter_fig

    # --- 5. Feature Distribution Histograms ---
    if data is not None and "X" in data:
        X_full = data["X"]
        y_full = data["y"]
        dist_df = pd.DataFrame(X_full, columns=feature_names)
        dist_df["Species"] = [le_classes[y] for y in y_full]

        # Select top 2 most important features for distribution
        top_2_feats = sorted(
            result["feature_importance"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:2]
        top_2_names = [f[0] for f in top_2_feats]

        for feat_name in top_2_names:
            if feat_name in dist_df.columns:
                hist_fig = px.histogram(
                    dist_df,
                    x=feat_name,
                    color="Species",
                    nbins=25,
                    barmode="overlay",
                    opacity=0.7,
                    title=f"Distribution of {feat_name} by Species",
                    color_discrete_sequence=px.colors.qualitative.Set2,
                )
                hist_fig.update_layout(height=400)
                figures[f"dist_{feat_name}"] = hist_fig

    return figures


# =============================================================================
# UI ENGINE
# =============================================================================

def render_problem_setup():
    """
    Render the Problem Setup Module in the sidebar.
    Provides dataset upload, default dataset selection, and input validation.
    """
    st.sidebar.header("1. Problem Setup")
    st.sidebar.markdown("---")

    st.sidebar.markdown("""
    **Task:** Classify Iris flowers into 3 species based on measurements.

    **Inputs:**
    - Sepal length & width (cm)
    - Petal length & width (cm)

    **Output:** Species (setosa, versicolor, virginica)

    **Constraints:**
    - Minimum 10 samples required
    - Last column must be the target label
    - Numeric features only
    """)

    st.sidebar.markdown("---")

    # Dataset source selection
    dataset_source = st.sidebar.radio(
        "Select Dataset Source:",
        options=["Use Default Dataset", "Upload My Own CSV"],
        index=0,
        help="Choose the built-in Iris dataset or upload your own classification data.",
    )

    data = None

    if dataset_source == "Upload My Own CSV":
        uploaded_file = st.sidebar.file_uploader(
            "Upload CSV file:",
            type=["csv"],
            help="CSV should have numeric features and the last column as the target label.",
        )
        if uploaded_file is not None:
            try:
                data = pd.read_csv(uploaded_file)
                st.sidebar.success(f"Uploaded: {uploaded_file.name} ({data.shape[0]} rows, {data.shape[1]} columns)")
            except Exception as e:
                st.sidebar.error(f"Failed to read uploaded file: {e}")
                data = None
        else:
            st.sidebar.info("Please upload a CSV file to proceed.")
    else:
        # Use default dataset
        default_path = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
        data = load_data(default_path)
        if data is not None:
            st.sidebar.success(f"Default Iris dataset loaded: {data.shape[0]} samples")

    return data


def render_visual_ui():
    """
    Render the Visual UI Module with beautiful controls:
    sliders for test-size split, dropdowns for algorithm/hyperparameters.

    Returns:
        dict: User-selected parameters, or None if not ready.
    """
    st.sidebar.header("2. Model Configuration")
    st.sidebar.markdown("---")

    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Select Algorithm:",
        options=["Logistic Regression", "Random Forest"],
        index=0,
        help="Choose the machine learning algorithm to train.",
    )

    # Test size slider
    test_size = st.sidebar.slider(
        "Test Set Split (%):",
        min_value=10,
        max_value=50,
        value=20,
        step=5,
        help="Percentage of data reserved for testing.",
    ) / 100.0

    # Random state
    random_state = st.sidebar.number_input(
        "Random State (for reproducibility):",
        min_value=0,
        max_value=1000,
        value=42,
        step=1,
    )

    # Hyperparameters based on algorithm
    params = {
        "algorithm": algorithm,
        "test_size": test_size,
        "random_state": random_state,
    }

    st.sidebar.markdown("---")
    st.sidebar.subheader("Hyperparameters")

    if algorithm == "Random Forest":
        params["n_estimators"] = st.sidebar.slider(
            "Number of Trees:",
            min_value=10,
            max_value=300,
            value=100,
            step=10,
            help="Number of decision trees in the forest.",
        )
        params["max_depth"] = st.sidebar.slider(
            "Max Tree Depth:",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            help="Maximum depth of each tree. Higher = more complex model.",
        )
        if params["max_depth"] >= 30:
            params["max_depth"] = None
    else:
        params["C"] = st.sidebar.slider(
            "Regularization Strength (C):",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.01,
            format="%.2f",
            help="Inverse of regularization strength. Lower = stronger regularization.",
        )

    # Run button
    st.sidebar.markdown("---")
    run_clicked = st.sidebar.button(
        "Run Experiment",
        type="primary",
        use_container_width=True,
    )

    if run_clicked:
        return params
    return None


def render_evaluation(result):
    """
    Render the Evaluation Module: display performance metrics and visualizations.

    Parameters:
        result (dict): Output from run_model_or_algorithm().
    """
    if result is None:
        return

    st.header("Experiment Results")
    st.markdown(f"**Algorithm:** {result['algorithm']} | **Timestamp:** {result['timestamp']}")
    st.markdown("---")

    # --- Key Metrics Cards ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Accuracy",
            value=f"{result['accuracy']:.2%}",
        )
    with col2:
        st.metric(
            label="Precision",
            value=f"{result['precision']:.2%}",
        )
    with col3:
        st.metric(
            label="Recall",
            value=f"{result['recall']:.2%}",
        )
    with col4:
        st.metric(
            label="Test Samples",
            value=f"{result['n_test']}",
        )

    st.markdown("---")

    # --- Visualizations ---
    figures = create_visuals(st.session_state.preprocessed, result)

    if figures:
        # Top row: Metrics gauge + Confusion matrix
        col_left, col_right = st.columns([1, 1])

        with col_left:
            if "metrics_gauge" in figures:
                st.plotly_chart(figures["metrics_gauge"], use_container_width=True, key="gauge")

        with col_right:
            if "confusion_matrix" in figures:
                st.plotly_chart(figures["confusion_matrix"], use_container_width=True, key="cm")

        # Middle row: Feature importance + Scatter plot
        col_left2, col_right2 = st.columns([1, 1])

        with col_left2:
            if "feature_importance" in figures:
                st.plotly_chart(figures["feature_importance"], use_container_width=True, key="fi")

        with col_right2:
            if "scatter_plot" in figures:
                st.plotly_chart(figures["scatter_plot"], use_container_width=True, key="scatter")

        # Bottom row: Distribution plots
        dist_keys = [k for k in figures.keys() if k.startswith("dist_")]
        if dist_keys:
            st.subheader("Feature Distributions")
            for dk in dist_keys:
                st.plotly_chart(figures[dk], use_container_width=True, key=f"dist_{dk}")

    st.markdown("---")

    # --- Explanation Section ---
    st.subheader("Model Explanation")
    explanation = generate_explanation(result)
    st.markdown(explanation)

    # --- Save to Comparison ---
    st.markdown("---")
    save_col1, save_col2 = st.columns([1, 3])
    with save_col1:
        if st.button("Save This Run", type="primary", use_container_width=True):
            # Check if we've reached the limit of 5 runs
            if len(st.session_state.run_history) >= 5:
                st.warning("Maximum 5 runs can be saved. Remove a run before saving a new one.")
            else:
                # Check if this exact config already exists
                run_entry = {
                    "run_id": len(st.session_state.run_history) + 1,
                    "algorithm": result["algorithm"],
                    "test_size": result["params"]["test_size"],
                    "accuracy": result["accuracy"],
                    "precision": result["precision"],
                    "recall": result["recall"],
                    "n_estimators": result["params"].get("n_estimators", "-"),
                    "max_depth": result["params"].get("max_depth", "-"),
                    "C": result["params"].get("C", "-"),
                    "random_state": result["params"]["random_state"],
                    "timestamp": result["timestamp"],
                }
                st.session_state.run_history.append(run_entry)
                st.success(f"Run #{run_entry['run_id']} saved to comparison table!")


def render_comparison_module():
    """
    Render the Evaluation & Comparison Module.
    Displays a history table comparing up to 5 saved experiment runs.
    """
    st.header("Experiment Comparison")
    st.markdown("Compare up to **5 different runs** to find the best configuration.")
    st.markdown("---")

    if not st.session_state.run_history:
        st.info("No saved experiments yet. Run an experiment and click **'Save This Run'** to add it here.")
        return

    # Build comparison DataFrame
    comparison_df = pd.DataFrame(st.session_state.run_history)

    # Format columns for display
    display_df = comparison_df.copy()
    display_df["accuracy"] = display_df["accuracy"].apply(lambda x: f"{x:.2%}")
    display_df["precision"] = display_df["precision"].apply(lambda x: f"{x:.2%}")
    display_df["recall"] = display_df["recall"].apply(lambda x: f"{x:.2%}")
    display_df["test_size"] = display_df["test_size"].apply(lambda x: f"{x:.0%}")

    # Reorder columns
    col_order = [
        "run_id", "algorithm", "test_size", "accuracy",
        "precision", "recall", "n_estimators", "max_depth", "C",
        "random_state", "timestamp",
    ]
    display_df = display_df[[c for c in col_order if c in display_df.columns]]

    # Rename columns nicely
    display_df.columns = [c.replace("_", " ").title() for c in display_df.columns]

    st.subheader("Saved Experiments")
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # --- Best Run Highlight ---
    best_idx = comparison_df["accuracy"].idxmax()
    best_run = comparison_df.iloc[best_idx]

    st.markdown("---")
    st.subheader("Best Performing Run")

    bcol1, bcol2, bcol3, bcol4 = st.columns(4)
    with bcol1:
        st.metric("Run ID", f"#{best_run['run_id']}")
    with bcol2:
        st.metric("Algorithm", best_run["algorithm"])
    with bcol3:
        st.metric("Accuracy", f"{best_run['accuracy']:.2%}")
    with bcol4:
        st.metric("Configuration", f"Test={best_run['test_size']:.0%}")

    # --- Accuracy Comparison Bar Chart ---
    st.markdown("---")
    st.subheader("Accuracy Comparison Chart")

    chart_df = comparison_df.copy()
    chart_df["run_label"] = chart_df.apply(
        lambda r: f"Run {r['run_id']}: {r['algorithm'][:3]}", axis=1
    )

    acc_fig = px.bar(
        chart_df,
        x="run_label",
        y="accuracy",
        color="algorithm",
        text=chart_df["accuracy"].apply(lambda x: f"{x:.2%}"),
        title="Accuracy Across Saved Runs",
        labels={"accuracy": "Accuracy", "run_label": "Experiment"},
        height=400,
    )
    acc_fig.update_traces(textposition="outside")
    acc_fig.update_layout(yaxis_range=[0, 1.05], showlegend=True)
    st.plotly_chart(acc_fig, use_container_width=True, key="acc_compare")

    # --- Remove runs ---
    st.markdown("---")
    st.subheader("Manage Saved Runs")
    runs_to_remove = st.multiselect(
        "Select runs to remove:",
        options=[f"Run {r['run_id']}: {r['algorithm']} ({r['timestamp']})" for r in st.session_state.run_history],
        help="Select one or more saved runs to delete from the comparison table.",
    )
    if st.button("Remove Selected Runs", type="secondary"):
        if runs_to_remove:
            ids_to_remove = [int(r.split(":")[0].replace("Run ", "")) for r in runs_to_remove]
            st.session_state.run_history = [
                r for r in st.session_state.run_history if r["run_id"] not in ids_to_remove
            ]
            # Reassign run IDs
            for i, run in enumerate(st.session_state.run_history):
                run["run_id"] = i + 1
            st.rerun()

    st.markdown("---")
    st.caption(f"Total saved runs: **{len(st.session_state.run_history)}** / 5 maximum")


def render_ui():
    """
    Main UI Engine: Render the complete Streamlit interface wrapping all modules together.
    """
    # Page configuration
    st.set_page_config(
        page_title="AI Classifier Lab | GroupAlpha",
        page_icon="🌸",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stMetric {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
    }
    div[data-testid="stButton"] > button {
        border-radius: 8px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    init_session_state()

    # --- HEADER ---
    st.markdown('<div class="main-header">🌸 AI Classifier Lab</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Interactive Machine Learning: Iris Flower Classification | GroupAlpha</div>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    # --- SIDEBAR MODULES ---
    st.sidebar.title("⚙️ Control Panel")
    st.sidebar.markdown("---")

    # 1. Problem Setup Module
    data = render_problem_setup()

    # Store data in session state
    if data is not None:
        st.session_state.data = data

    st.sidebar.markdown("---")

    # Only show model config if data is loaded
    params = None
    if st.session_state.data is not None:
        # 2. Visual UI Module
        params = render_visual_ui()
    else:
        st.sidebar.warning("Load a dataset to configure the model.")

    # --- MAIN CONTENT AREA ---
    if st.session_state.data is not None:
        # Display dataset preview
        with st.expander("📊 Dataset Preview", expanded=False):
            st.subheader("Dataset Overview")
            st.dataframe(st.session_state.data.head(15), use_container_width=True)
            st.caption(f"Shape: {st.session_state.data.shape}")

            # Show basic stats
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Feature Statistics:**")
                st.dataframe(st.session_state.data.describe(), use_container_width=True)
            with col_b:
                st.markdown("**Class Distribution:**")
                target_col = st.session_state.data.columns[-1]
                dist = st.session_state.data[target_col].value_counts()
                st.dataframe(dist, use_container_width=True)

        # Preprocess data
        if st.session_state.preprocessed is None:
            with st.spinner("Preprocessing data..."):
                st.session_state.preprocessed = preprocess_data(st.session_state.data)

        # Run experiment if button clicked
        if params is not None and st.session_state.preprocessed is not None:
            result = run_model_or_algorithm(st.session_state.preprocessed, params)
            st.session_state.current_result = result

            if result is not None:
                st.success("Experiment completed successfully!")

        # Display results if available
        if st.session_state.current_result is not None:
            render_evaluation(st.session_state.current_result)

        # --- COMPARISON MODULE ---
        st.markdown("---")
        render_comparison_module()

    else:
        # No data loaded yet
        st.info("👈 Please configure your dataset in the sidebar to get started.")

        # Show placeholder image / description
        st.markdown("""
        ### Welcome to the AI Classifier Lab!

        This interactive application allows you to:

        1. **Load Data** — Use the built-in Iris dataset or upload your own CSV file
        2. **Configure Model** — Select algorithms and tune hyperparameters via the sidebar
        3. **Train & Evaluate** — Run experiments and view performance metrics in real-time
        4. **Visualize** — Explore interactive Plotly charts including confusion matrices and feature importance
        5. **Compare** — Save up to 5 runs and compare which configuration performs best

        **The Iris Dataset** contains 150 samples of iris flowers with 4 numerical features:
        sepal length, sepal width, petal length, and petal width. The goal is to classify
        each flower into one of three species: *setosa*, *versicolor*, or *virginica*.

        Get started by selecting a dataset source in the sidebar! 🚀
        """)

    # --- FOOTER ---
    st.markdown("---")
    st.caption("AI Classifier Lab | GroupAlpha | Built with Streamlit + scikit-learn + Plotly")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    render_ui()
