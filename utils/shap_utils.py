import shap
import matplotlib.pyplot as plt

def plot_shap_waterfall(model, features):
    explainer = shap.Explainer(model)
    shap_values = explainer(features)
    fig = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    return fig



