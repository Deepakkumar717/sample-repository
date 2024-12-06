import pymysql

# Function to insert criminal data into the database
def insertData(data):
    rowId = 0
    # Debugging print statement: Print key-value pairs
    print("Entry Data (key-value pairs):")
    for key, value in data.items():
        print(f"{key}: {value}")

    # Connect to the MySQL database
    db = pymysql.connect(host="localhost", user="root", password="Mysql#rjd22004", database="criminal")
    cursor = db.cursor()
    print("database connected")

    # SQL query to insert data into the criminaldata table
    query = "INSERT INTO criminaldata1 (Name, Father_Name, Gender, DOB, Crimes_Done) VALUES ('%s', '%s', '%s', '%s', '%s');" % \
            (data["Name"], data["Father_Name"], data["Gender"],
             data["DOB(yyyy-mm-dd)"], data["Crimes_Done"])

    try:
        # Execute the SQL query
        cursor.execute(query)
        # Commit the transaction
        db.commit()
        # Get the last inserted row ID
        rowId = cursor.lastrowid
        print("data stored on row %d" % rowId)
    except Exception as e:
        # Rollback the transaction if any error occurs
        db.rollback()
        print(f"Data insertion failed: {e}")

    # Close the database connection
    db.close()
    print("connection closed")
    return rowId

# Function to retrieve criminal data based on the name
def retrieveData(name):
    id = None
    crim_data = None

    # Connect to the MySQL database
    db = pymysql.connect(host="localhost", user="root", password="Mysql#rjd22004", database="criminal")
    cursor = db.cursor()
    print("database connected")

    # SQL query to select the data based on the criminal's name
    query = "SELECT id, Name, Father_name, Gender, DOB, Crimes_Done FROM criminaldata1 WHERE name='%s'" % name

    try:
        # Execute the SQL query
        cursor.execute(query)
        # Fetch the first result
        result = cursor.fetchone()

        # If data is found, extract it
        if result:
            id = result[0]
            crim_data = {
                "Name": result[1],
                "Father_Name": result[2],
                "Gender": result[3],
                "DOB(yyyy-mm-dd)": result[4],
                "Crimes_Done": result[5]
            }
            print("data retrieved")
        else:
            print("No data found for the given name")
    except Exception as e:
        print(f"Error: Unable to fetch data: {e}")

    # Close the database connection
    db.close()
    print("connection closed")

    return id, crim_data
