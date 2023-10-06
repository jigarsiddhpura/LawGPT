from dotenv import load_dotenv
load_dotenv()

# with openai
OPENAI_API_KEY = os.getenv("OPENAI_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY