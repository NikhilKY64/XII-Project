import mysql.connector
from tabulate import tabulate

conn = mysql.connector.connect(
    host='localhost',
    user='root',
    password='admin'
)

cursor = conn.cursor()

cursor.execute("CREATE DATABASE IF NOT EXISTS CRMS")
conn.database = 'CRMS'

cursor.execute("""CREATE TABLE IF NOT EXISTS students(
        id int AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(50),
        age INT,
        attendance ENUM('present', 'absent') NOT NULL
)""")

while True:

    print("Options")
    print("1) Insert_data")
    print("2) Update_data")
    print("3) Delete_data")
    print("4) Show_table")
    print("5) Exit")


    option = int(input("Enter your option: "))


    if option == 1:
        name = input("Enter your name: ")
        age = int(input("Enter your age: "))
        attendance = input("Enter present or absent: ")
        cursor.execute("INSERT INTO students (name, age, attendance) VALUES (%s,%s,%s)",(name, age, attendance))
        conn.commit()
        print('Data inserted successfully.')

    elif option == 2:
        name = input("Enter your name: ")
        age = int(input("Enter your age: "))
        attendance = input("Enter present or absent: ")
        cursor.execute("UPDATE students SET age=%s, attendance=%s WHERE name=%s", (age, attendance, name))
        conn.commit()
        print("Data updated successfully.")

    elif option == 3:
        name = input("Enter your name: ")
        age = int(input("Enter your age: "))
        attendance = input("Enter present or absent: ")
        cursor.execute("DELETE FROM students WHERE name=%s", (name,))
        conn.commit()
        print("Data deleted successfully.")

    elif option == 4:

        name = input("Enter your name: ")
        cursor.execute("SELECT * FROM students WHERE name=%s", (name,))
        data = cursor.fetchall()
        if data:
            print("Data found!")
            print(tabulate(data, headers=["id", "name", "age", "attendance"], tablefmt="rounded_grid"))
            # table format options: "plain", "grid", "heavy_grid", "double_grid", "rounded_grid", etc.

        else:
            print("Data not found!")

    elif option == 5:
        print("Exiting the program...")
        break

    else:
        print("Invalid option. Please try again.")
    
    print("Do you want to continue? (yes/no)")

    continue_option = input().lower()
    if continue_option != "yes" and continue_option != "y":
        print("Exiting the program...")
        break   
