import logging

logging.basicConfig(
    level=logging.INFO,  # Minimum log level to handle
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    handlers=[
        logging.FileHandler("app.log"),  # Log to file
        logging.StreamHandler(),  # Also log to console
    ],
)

logger = logging.getLogger()


def main():
    pass


if __name__ == "__main__":
    pass
