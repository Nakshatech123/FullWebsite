# from fastapi import APIRouter, HTTPException, status, Request, Depends
# from pydantic import BaseModel, EmailStr, constr
# from passlib.hash import bcrypt
# import sqlite3
# from datetime import datetime, timedelta
# from jose import JWTError, jwt

# router = APIRouter()

# # JWT Settings
# JWT_SECRET_KEY = "naksha-secret-key"
# JWT_ALGORITHM = "HS256"
# JWT_ACCESS_EXPIRE_MINUTES = 60
# JWT_REFRESH_EXPIRE_DAYS = 7

# # Shared internal credential
# SHARED_SECRET = "Nakshatech"

# # User input schema
# class UserAuth(BaseModel):
#     email: EmailStr
#     password: constr(min_length=6)
#     secret: str

# class RefreshRequest(BaseModel):
#     refresh_token: str

# # DB setup
# def create_user_table():
#     with sqlite3.connect("users.db") as conn:
#         conn.execute(
#             "CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, password TEXT)"
#         )

# create_user_table()

# # Token generators
# def create_access_token(data: dict, expires_delta: timedelta = None):
#     expire = datetime.utcnow() + (expires_delta or timedelta(minutes=JWT_ACCESS_EXPIRE_MINUTES))
#     data.update({"exp": expire})
#     return jwt.encode(data, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

# def create_refresh_token(data: dict):
#     expire = datetime.utcnow() + timedelta(days=JWT_REFRESH_EXPIRE_DAYS)
#     data.update({"exp": expire})
#     return jwt.encode(data, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

# # Routes
# @router.post("/register")
# def register(user: UserAuth):
#     if user.secret != SHARED_SECRET:
#         raise HTTPException(status_code=401, detail="❌ Invalid shared key.")
    
#     hashed_pw = bcrypt.hash(user.password)
#     try:
#         with sqlite3.connect("users.db") as conn:
#             conn.execute("INSERT INTO users (email, password) VALUES (?, ?)", (user.email, hashed_pw))
#     except sqlite3.IntegrityError:
#         raise HTTPException(status_code=400, detail=f"❌ User already exists.")
    
#     return {
#         "status": "success",
#         "message": "✅ Registered successfully.",
#         "email": user.email
#     }

# @router.post("/login")
# def login(user: UserAuth):
#     if user.secret != SHARED_SECRET:
#         raise HTTPException(status_code=401, detail="❌ Invalid shared key.")
    
#     with sqlite3.connect("users.db") as conn:
#         cur = conn.cursor()
#         cur.execute("SELECT password FROM users WHERE email = ?", (user.email,))
#         record = cur.fetchone()

#         if not record or not bcrypt.verify(user.password, record[0]):
#             raise HTTPException(status_code=401, detail="❌ Invalid email or password.")

#     token_data = {"sub": user.email}
#     access_token = create_access_token(data=token_data)
#     refresh_token = create_refresh_token(data=token_data)

#     return {
#         "status": "success",
#         "message": "✅ Login successful.",
#         "email": user.email,
#         "access_token": access_token,
#         "refresh_token": refresh_token,
#         "token_type": "bearer"
#     }

# @router.post("/refresh")
# def refresh_token(payload: RefreshRequest):
#     try:
#         decoded = jwt.decode(payload.refresh_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
#         email = decoded.get("sub")
#         if not email:
#             raise HTTPException(status_code=401, detail="❌ Invalid refresh token.")
#     except JWTError:
#         raise HTTPException(status_code=401, detail="❌ Expired or invalid refresh token.")
    
#     new_token = create_access_token(data={"sub": email})
#     return {
#         "status": "success",
#         "access_token": new_token,
#         "token_type": "bearer"
#     }





from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr, constr
from passlib.hash import bcrypt
import sqlite3

# Import the utility functions from the file in the Canvas
from auth_utils import create_access_token, get_current_user

router = APIRouter()

# --- Configuration ---
SHARED_SECRET = "Nakshatech"
DB_PATH = "users.db"

# --- Pydantic Models ---
class UserAuth(BaseModel):
    email: EmailStr
    password: constr(min_length=6)
    secret: str

# --- Database Setup ---
def create_user_table():
    """Ensures the users table exists in the database."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, password TEXT)"
        )
create_user_table()

# --- API Routes ---

@router.post("/register")
def register(user: UserAuth):
    """Handles new user registration."""
    if user.secret != SHARED_SECRET:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid shared key.")
    
    hashed_pw = bcrypt.hash(user.password)
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO users (email, password) VALUES (?, ?)", (user.email, hashed_pw))
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="User with this email already exists.")
    
    return {"status": "success", "message": "Registered successfully."}

@router.post("/login")
def login(user: UserAuth):
    """Handles user login and returns an access token."""
    if user.secret != SHARED_SECRET:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid shared key.")
    
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        cur.execute("SELECT password FROM users WHERE email = ?", (user.email,))
        record = cur.fetchone()

        if not record or not bcrypt.verify(user.password, record[0]):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password.")

    access_token = create_access_token(data={"sub": user.email})

    return {
        "status": "success",
        "message": "Login successful.",
        "access_token": access_token,
        "token_type": "bearer"
    }

@router.get("/me")
def read_users_me(current_user: str = Depends(get_current_user)):
    """A protected route to validate a token and get the current user's email."""
    return {"email": current_user}


