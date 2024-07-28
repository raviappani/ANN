import streamlit as st
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

def load_data(filepath):
    df = pd.read_csv(filepath, header=None)
    x = df.drop(2, axis=1).values
    y = df[2].values.astype(int)
    return x, y

def build_model(input_features, hl_1, hl_2, ol, af_1, af_2, ol_3):
    input_layer = Input(shape=(input_features,))
    hidden_layer1 = Dense(hl_1, activation=af_1)(input_layer)
    hidden_layer2 = Dense(hl_2, activation=af_2)(hidden_layer1)
    output_layer = Dense(ol, activation=ol_3)(hidden_layer2)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def train_model(model, X_train, y_train, epochs):
    model.compile(optimizer=SGD(), loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=100, validation_split=0.2, verbose=0)
    return history

def plot_decision_region_for_neuron(x, y, model, title):
    plot_decision_regions(x, y, clf=model, legend=2)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend(loc='upper left')
    st.pyplot(plt.gcf())
    plt.clf()

def plot_history(history):
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(plt.gcf())
    plt.clf()

def main():
    st.title("Neural Network Decision Boundary Visualization")

    data_choice = st.sidebar.selectbox("Select Data Type", 
                                       ["ushape", "concentric", "linear", "xor", "spiral", "random", "outlier", "overlap"])

    filepath_dict = {
        "ushape": "C:\\Users\\ravinder\\Desktop\\DL\\multiple_csv\\u_shape_generated_points.csv",
        "concentric": "C:\\Users\\ravinder\\Desktop\\DL\\multiple_csv\\concentric_generated_points.csv",
        "linear": "C:\\Users\\ravinder\\Desktop\\DL\\multiple_csv\\linear_generated_points.csv",
        "xor": "C:\\Users\\ravinder\\Desktop\\DL\\multiple_csv\\xor_generated_points.csv",
        "spiral": "C:\\Users\\ravinder\\Desktop\\DL\\multiple_csv\\two_spirals_generated_points.csv",
        "random": "C:\\Users\\ravinder\\Desktop\\DL\\multiple_csv\\random_generated_points.csv",
        "outlier": "C:\\Users\\ravinder\\Desktop\\DL\\multiple_csv\\linear_outlier_generated_points.csv",
        "overlap": "C:\\Users\\ravinder\\Desktop\\DL\\multiple_csv\\linear_overlap_generated_points.csv"
    }
    
    selected_filepath = filepath_dict[data_choice]
    x, y = load_data(selected_filepath)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    st.sidebar.header("Neural Network Parameters")
    input_features = 2
    hl_1 = st.sidebar.number_input("Neurons in 1st Hidden Layer", min_value=1, max_value=100, value=10)
    hl_2 = st.sidebar.number_input("Neurons in 2nd Hidden Layer", min_value=1, max_value=100, value=10)
    ol = 1
    af_1 = st.sidebar.text_input("Activation Function in 1st Hidden Layer", value="relu")
    af_2 = st.sidebar.text_input("Activation Function in 2nd Hidden Layer", value="relu")
    ol_3 = st.sidebar.text_input("Activation Function in Output Layer", value="sigmoid")
    epochs = st.sidebar.number_input("Number of Epochs", min_value=1, max_value=1000, value=100)

    if st.sidebar.button("Train Model"):
        model = build_model(input_features, hl_1, hl_2, ol, af_1, af_2, ol_3)
        history = train_model(model, X_train, y_train, epochs)
        st.sidebar.success("Model Trained Successfully!")

        for j in range(hl_1):
            intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[1].output[:, j])
            plot_decision_region_for_neuron(x, y, intermediate_layer_model, f'Decision Region for Neuron {j+1} in First Hidden Layer')

        for j in range(hl_2):
            intermediate_layer_model = Model(inputs=model.input, outputs=model.layers[2].output[:, j])
            plot_decision_region_for_neuron(x, y, intermediate_layer_model, f'Decision Region for Neuron {j+1} in Second Hidden Layer')

        output_layer_model = Model(inputs=model.input, outputs=model.layers[-1].output)
        plot_decision_region_for_neuron(x, y, output_layer_model, "Output Layer")

        plot_history(history)

if __name__ == "__main__":
    main()
