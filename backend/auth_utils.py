# # auth_utils.py
# from fastapi import Depends, HTTPException, status
# from fastapi.security import APIKeyHeader
# from jose import JWTError, jwt
# from typing import Optional

# # JWT config
# JWT_SECRET_KEY = "naksha-secret-key"
# JWT_ALGORITHM = "HS256"

# # Define the header we expect in Swagger / requests
# auth_header = APIKeyHeader(name="Authorization")

# # Token decoder
# def get_current_user(token: str = Depends(auth_header)) -> str:
#     try:
#         # Strip "Bearer " prefix if present
#         if token.startswith("Bearer "):
#             token = token[7:]

#         payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
#         email: Optional[str] = payload.get("sub")

#         if email is None:
#             raise HTTPException(
#                 status_code=status.HTTP_401_UNAUTHORIZED,
#                 detail="❌ Invalid token payload.",
#                 headers={"WWW-Authenticate": "Bearer"},
#             )

#         return email

#     except JWTError:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="❌ Invalid or expired token.",
#             headers={"WWW-Authenticate": "Bearer"},
#         )



from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from datetime import datetime, timedelta
from jose import JWTError, jwt

# --- Configuration ---
JWT_SECRET_KEY = "naksha-secret-key"
JWT_ALGORITHM = "HS256"
JWT_ACCESS_EXPIRE_MINUTES = 60

# This scheme points to the login endpoint and tells FastAPI how to find the token.
# It's the mechanism that makes the "Authorize" button work in Swagger UI.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def create_access_token(data: dict):
    """Creates a new JWT access token."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=JWT_ACCESS_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)):
    """
    Decodes the JWT token to get the current user's email.
    This is the dependency function used to protect all secure API routes.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    return email


