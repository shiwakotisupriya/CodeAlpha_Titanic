import tkinter as tk
from tkinter import messagebox
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score


svm_model = joblib.load('svm_model.joblib')
tree_model = joblib.load('decision_tree_model.joblib')
log_reg_model = joblib.load('logistic_reg.joblib')

def convert_prediction(prediction):
    return "Yes" if prediction == 1 else "No"


def predict_survival(fare, embarked):

    user_input = pd.DataFrame([[fare, embarked]], columns=['Fare', 'Embarked'])

    svm_prediction = convert_prediction(svm_model.predict(user_input)[0])
    tree_prediction = convert_prediction(tree_model.predict(user_input)[0])
    log_reg_prediction = convert_prediction(log_reg_model.predict(user_input)[0])

 
    result_text = f"SVM Prediction: {svm_prediction}\n" \
                  f"Decision Tree Prediction: {tree_prediction}\n" \
                  f"Logistic Regression Prediction: {log_reg_prediction}"
    result_label.config(text=result_text)


def show_accuracy(X_test, y_test):
  
    svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))
    tree_accuracy = accuracy_score(y_test, tree_model.predict(X_test))
    log_reg_accuracy = accuracy_score(y_test, log_reg_model.predict(X_test))

   
    acc_text = f"SVM Accuracy: {svm_accuracy:.2f}\n" \
               f"Decision Tree Accuracy: {tree_accuracy:.2f}\n" \
               f"Logistic Regression Accuracy: {log_reg_accuracy:.2f}"
    messagebox.showinfo("Model Accuracy", acc_text)


root = tk.Tk()
root.title("Titanic Survival Prediction")
root.attributes('-fullscreen', True) 


def exit_fullscreen(event):
    root.attributes('-fullscreen', False)

root.bind("<Escape>", exit_fullscreen)


label_font = ("Arial", 16)
entry_font = ("Arial", 14)
button_font = ("Arial", 14)


form_frame = tk.Frame(root)
form_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)


fare_label = tk.Label(form_frame, text="Fare:", font=label_font)
fare_label.grid(row=0, column=0, pady=10, padx=10)
fare_entry = tk.Entry(form_frame, font=entry_font, width=10)
fare_entry.grid(row=0, column=1, pady=10, padx=10)


embarked_label = tk.Label(form_frame, text="Embarked (0=C, 1=Q, 2=S):", font=label_font)
embarked_label.grid(row=1, column=0, pady=10, padx=10)
embarked_entry = tk.Entry(form_frame, font=entry_font, width=10)
embarked_entry.grid(row=1, column=1, pady=10, padx=10)


result_label = tk.Label(form_frame, text="", font=label_font)
result_label.grid(row=3, column=0, columnspan=2, pady=20)


def on_predict():
    try:
        fare = float(fare_entry.get())
        embarked = int(embarked_entry.get())
        predict_survival(fare, embarked)
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter valid numerical values.")

predict_button = tk.Button(form_frame, text="Predict Survival", command=on_predict, font=button_font)
predict_button.grid(row=2, column=0, columnspan=2, pady=10)


def on_accuracy():
  
    X_test = pd.DataFrame([[1.0, 0], [2.5, 1], [3.0, 2]], columns=['Fare', 'Embarked'])
    y_test = pd.Series([0, 1, 0]) 
    show_accuracy(X_test, y_test)

accuracy_button = tk.Button(form_frame, text="Show Model Accuracy", command=on_accuracy, font=button_font)
accuracy_button.grid(row=4, column=0, columnspan=2, pady=10)


root.mainloop()
