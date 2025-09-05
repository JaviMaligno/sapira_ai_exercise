#!/usr/bin/env python3
"""
Generate MLflow Authentication Password Hashes
==============================================

This script generates bcrypt hashes for MLflow basic authentication.
"""

import bcrypt

def generate_password_hash(password: str) -> str:
    """Generate bcrypt hash for a password."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def main():
    """Generate password hashes for MLflow users."""
    
    users = {
        "admin": "mlflow-admin-2025",
        "analyst": "fraud-analyst-secure", 
        "scientist": "ml-scientist-pass"
    }
    
    print("MLflow Password Hashes")
    print("=" * 30)
    
    for username, password in users.items():
        hash_value = generate_password_hash(password)
        print(f"\n{username}:")
        print(f"  Password: {password}")
        print(f"  Hash: {hash_value}")

if __name__ == "__main__":
    main()