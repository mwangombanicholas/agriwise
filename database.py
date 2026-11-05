import sqlite3
import bcrypt

def init_db():
    conn = sqlite3.connect('agriwise.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS farmers 
                 (id INTEGER PRIMARY KEY, name TEXT, email TEXT UNIQUE, password TEXT)''')
    conn.commit()
    conn.close()

def add_farmer(name, email, password):
    conn = sqlite3.connect('agriwise.db')
    c = conn.cursor()
    # Hash the password for security
    hashed_pw = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    try:
        c.execute("INSERT INTO farmers (name, email, password) VALUES (?, ?, ?)", (name, email, hashed_pw))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # Email already exists
    finally:
        conn.close()

def verify_farmer(email, password):
    conn = sqlite3.connect('agriwise.db')
    c = conn.cursor()
    c.execute("SELECT name, password FROM farmers WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    
    if user:
        stored_name, stored_hash = user
        # Check if the password matches the hash
        if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
            return stored_name
    return None
