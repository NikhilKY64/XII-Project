import mysql.connector
from tabulate import tabulate
from datetime import date

connt = mysql.connector.connect(
    host="localhost",
    user="root",
    password="admin")

cur = connt.cursor()

cur.execute("create database if not exists cams")
connt.database="cams"

cur.execute("""create table if not exists students(
            roll_no INT primary key,
            name varchar(50) not null,
            attendance enum('present', 'absent') NOT NULL,
            attendance_date DATE NOT NULL
            )""")

while True:

    print("Options")
    print("1) Insert_data")
    print("2) Update_data")
    print("3) Delete_data")
    print("4) Show_table")
    print("5) Custom")
    print("6) Exit")


    option = int(input("Enter your option: "))


    if option == 1:
        while True:
            roll_no = int(input("Enter your Roll_No: "))
            name = input("Enter your name: ")
            attendance = input("Enter present or absent: ")
            attendance_date = date.today()
            cur.execute("INSERT INTO students (roll_no, name, attendance, attendance_date) VALUES (%s,%s,%s,%s)",(roll_no, name, attendance, attendance_date))
            connt.commit()
            more = input("Add another student? (y/n): ")
            if more.lower() != 'y':
                break

    elif option == 2:
        roll_no = int(input("Enter your Roll_No: "))
        name = input("Enter your name: ")
        attendance = input("Enter present or absent: ")
        cur.execute("UPDATE students SET name=%s, attendance=%s WHERE roll_no=%s",(name, attendance, roll_no))
        connt.commit()

    elif option == 3:
        roll_no = int(input("Enter your Roll_No to delete: "))
        cur.execute("DELETE FROM students WHERE roll_no=%s",(roll_no,))
        connt.commit()

    elif option == 4:
        cur.execute("SELECT * FROM students ORDER BY roll_no")
        records = cur.fetchall()
        if records:
            print("Data Found!")
            print("Students Table:")
            print(tabulate(records, headers=["roll_no", "name", "attendance", "attendance_date"], tablefmt='rounded_grid'))
        else:
            print("No records found.")
            
    elif option == 5:
        user_input = input("Enter you custom query: ")
        cur.execute(user_input)

        if user_input.strip().upper().startswith('SELECT'):
            results = cur.fetchall()
            print(tabulate(results,headers=["roll_no", "name", "attendance", "attendance_date"],tablefmt="rounded_grid"))
        else:
            connt.commit()

    elif option == 6:
        print("Exiting the program...")
        connt.close()
        break

    else:
        print("Invalid option. Please try again.")


    print("Do you want to continue in programn?")
    user_inp = input("Enter (y/n): ").lower()

    if user_inp != 'y':
        print("Bye!")
        break