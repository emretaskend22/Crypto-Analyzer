# main.py
import uvicorn
from app.server import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "app.server:create_app",  # path to the factory function
        host="127.0.0.1",
        port=8000,
        reload=True,
        factory=True  # use create_app() factory
    )
