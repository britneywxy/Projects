import tkinter as tk
from tkinter import scrolledtext, messagebox

class TaskApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Task Entry")

        tk.Label(self.root, text="Enter your tasks:").pack(pady=10)

        self.task_text_box = scrolledtext.ScrolledText(self.root, width=50, height=20, font=("Arial", 30))
        self.task_text_box.pack(pady=5)

        submit_button = tk.Button(self.root, text="Submit Tasks", command=self.submit_tasks)
        submit_button.pack(pady=10)

        self.input_text = None  # To hold the input text once submitted

    def submit_tasks(self):
        input_text = self.task_text_box.get('1.0', tk.END).strip()  # Fetch text from text box
        if input_text:
            self.input_text = input_text  # Store the input text in the instance variable
            # messagebox.showinfo("Tasks Submitted", "Tasks have been submitted successfully.")
            self.root.quit()  # This will break the mainloop without destroying the window
        else:
            messagebox.showerror("Error", "Please enter some tasks before submitting.")

    def run(self):
        self.root.mainloop()
        self.root.destroy()  # Destroy the window after the mainloop ends
        return self.input_text  # Return the input text after the GUI has been closed


def create_main_window():
    root = tk.Tk()
    app = TaskApp(root)
    return app.run()  # This will start the GUI and return the input text after it is submitted

# if __name__ == "__main__":
#     input_text = create_main_window()
#     print(input_text)  # Do something with the input_text here
