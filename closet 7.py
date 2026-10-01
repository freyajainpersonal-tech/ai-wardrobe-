import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import json
import uuid
import requests
from datetime import datetime
import sys
from io import BytesIO


# ============================================================
# OPTIONAL HUGGING FACE SUPPORT
# ============================================================

try:
    from gradio_client import Client, handle_file
except ImportError:
    Client = None
    handle_file = None


# ============================================================
# PYTHON CHECK
# ============================================================

if sys.version_info < (3, 10):
    st.error("Python 3.10 or newer is required.")
    st.stop()


# ============================================================
# APP SETTINGS
# ============================================================

APP_TITLE = "AI Wardrobe Stylist"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="👗",
    layout="wide"
)


# ============================================================
# DATA DIRECTORIES
# ============================================================

DATA_DIR = "wardrobe_data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
DATA_FILE = os.path.join(DATA_DIR, "wardrobe.json")

PROFILE_IMAGE_DIR = os.path.join(DATA_DIR, "profile")

USER_PHOTO_FILE = os.path.join(
    PROFILE_IMAGE_DIR,
    "user_photo.jpg"
)

PROFILE_FILE = os.path.join(
    DATA_DIR,
    "profile.json"
)

os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(PROFILE_IMAGE_DIR, exist_ok=True)


# ============================================================
# HUGGING FACE / FASHN SETTINGS
# ============================================================

FASHN_SPACE = "fashn-ai/fashn-vton-1.5"


def get_huggingface_token():
    """
    Get the Hugging Face token securely.

    Priority:
    1. Streamlit secrets
    2. HF_TOKEN environment variable
    """

    # Try Streamlit secrets first.
    try:
        token = st.secrets.get("HF_TOKEN", "")

        if token:
            return str(token).strip()

    except Exception:
        pass

    # Environment variable fallback.
    token = os.getenv("HF_TOKEN", "")

    return token.strip()


# ============================================================
# AESTHETICS
# ============================================================

AESTHETICS = [
    "Cottagecore",
    "Fairycore",
    "Light Academia",
    "Dark Academia",
    "Clean Girl",
    "Cyberpunk",
    "Y2K",
    "Vaporwave",
    "Goblin Core",
    "Old Money",
    "Goth",
    "Mid-Century",
    "Indie",
    "Minimalist",
    "Kawaii",
    "Lolita",
    "Boho"
]


# ============================================================
# CLOTHING CATEGORIES
# ============================================================

CATEGORIES = [
    "Top",
    "T-Shirt",
    "Shirt",
    "Blouse",
    "Sweater",
    "Cardigan",
    "Hoodie",
    "Jacket",
    "Coat",
    "Dress",
    "Skirt",
    "Pants",
    "Jeans",
    "Shorts",
    "Shoes",
    "Bag",
    "Accessory",
    "Other"
]


# ============================================================
# AESTHETIC RULES
# ============================================================

AESTHETIC_RULES = {

    "Cottagecore": {
        "colors": ["Green", "White", "Yellow", "Pink", "Brown"],
        "categories": ["Dress", "Blouse", "Cardigan", "Skirt"],
        "keywords": [
            "floral", "linen", "lace",
            "cottage", "romantic", "nature"
        ]
    },

    "Fairycore": {
        "colors": ["Pink", "Purple", "Green", "White", "Cyan"],
        "categories": ["Dress", "Skirt", "Blouse", "Cardigan"],
        "keywords": [
            "fairy", "sparkle", "sheer",
            "floral", "wings"
        ]
    },

    "Light Academia": {
        "colors": ["White", "Brown", "Beige", "Gray", "Yellow"],
        "categories": [
            "Shirt", "Blouse", "Sweater",
            "Cardigan", "Pants", "Skirt"
        ],
        "keywords": [
            "academic", "classic", "knit",
            "preppy", "vintage"
        ]
    },

    "Dark Academia": {
        "colors": ["Black", "Brown", "Gray", "White", "Green"],
        "categories": [
            "Shirt", "Sweater", "Cardigan",
            "Jacket", "Pants", "Skirt"
        ],
        "keywords": [
            "academic", "tweed", "classic",
            "vintage", "dark"
        ]
    },

    "Clean Girl": {
        "colors": ["White", "Black", "Gray", "Beige"],
        "categories": [
            "T-Shirt", "Blouse",
            "Pants", "Jeans", "Cardigan"
        ],
        "keywords": [
            "minimal", "simple",
            "basic", "clean"
        ]
    },

    "Cyberpunk": {
        "colors": ["Black", "Purple", "Blue", "Cyan", "Red"],
        "categories": [
            "Hoodie", "Jacket",
            "Pants", "T-Shirt"
        ],
        "keywords": [
            "cyber", "tech",
            "neon", "futuristic", "utility"
        ]
    },

    "Y2K": {
        "colors": ["Pink", "Blue", "Purple", "White"],
        "categories": [
            "T-Shirt", "Top",
            "Skirt", "Jeans", "Pants"
        ],
        "keywords": [
            "y2k", "2000s",
            "baby tee", "denim", "retro"
        ]
    },

    "Vaporwave": {
        "colors": ["Pink", "Purple", "Cyan", "Blue"],
        "categories": [
            "T-Shirt", "Hoodie", "Top"
        ],
        "keywords": [
            "vaporwave", "neon",
            "retro", "digital"
        ]
    },

    "Goblin Core": {
        "colors": ["Green", "Brown", "Gray"],
        "categories": [
            "Sweater", "Hoodie",
            "Pants", "Jacket"
        ],
        "keywords": [
            "moss", "forest",
            "nature", "earth", "goblin"
        ]
    },

    "Old Money": {
        "colors": ["White", "Black", "Brown", "Beige", "Navy"],
        "categories": [
            "Blouse", "Shirt",
            "Cardigan", "Coat",
            "Pants", "Skirt"
        ],
        "keywords": [
            "classic", "tailored",
            "preppy", "elegant"
        ]
    },

    "Goth": {
        "colors": ["Black", "Purple", "Red", "Gray"],
        "categories": [
            "Dress", "T-Shirt",
            "Hoodie", "Jacket", "Skirt"
        ],
        "keywords": [
            "goth", "dark",
            "lace", "alternative"
        ]
    },

    "Mid-Century": {
        "colors": ["Brown", "Yellow", "Green", "Orange"],
        "categories": [
            "Dress", "Blouse",
            "Shirt", "Skirt", "Pants"
        ],
        "keywords": [
            "retro", "vintage",
            "mid century"
        ]
    },

    "Indie": {
        "colors": ["Green", "Brown", "Black", "Red"],
        "categories": [
            "T-Shirt", "Sweater",
            "Hoodie", "Jeans", "Jacket"
        ],
        "keywords": [
            "indie", "band",
            "vintage", "alternative"
        ]
    },

    "Minimalist": {
        "colors": ["White", "Black", "Gray", "Beige"],
        "categories": [
            "T-Shirt", "Shirt",
            "Blouse", "Pants", "Jeans"
        ],
        "keywords": [
            "minimal", "simple", "basic"
        ]
    },

    "Kawaii": {
        "colors": ["Pink", "White", "Purple", "Cyan"],
        "categories": [
            "Top", "Blouse",
            "Skirt", "Dress", "Cardigan"
        ],
        "keywords": [
            "kawaii", "cute",
            "pastel", "character"
        ]
    },

    "Lolita": {
        "colors": ["Black", "White", "Pink", "Red"],
        "categories": [
            "Dress", "Skirt",
            "Blouse", "Cardigan"
        ],
        "keywords": [
            "lolita", "lace",
            "ruffle", "bow"
        ]
    },

    "Boho": {
        "colors": ["Brown", "White", "Green", "Orange"],
        "categories": [
            "Dress", "Blouse",
            "Skirt", "Cardigan"
        ],
        "keywords": [
            "boho", "bohemian",
            "fringe", "embroidered"
        ]
    }
}


# ============================================================
# LOAD / SAVE WARDROBE
# ============================================================

def load_wardrobe():

    if not os.path.exists(DATA_FILE):
        return []

    try:
        with open(
            DATA_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if isinstance(data, list):
            return data

        return []

    except Exception:
        return []


def save_wardrobe(wardrobe):

    with open(
        DATA_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            wardrobe,
            f,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# LOAD / SAVE PROFILE
# ============================================================

def load_profile():

    if not os.path.exists(PROFILE_FILE):
        return {}

    try:

        with open(
            PROFILE_FILE,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(f)

        if isinstance(data, dict):
            return data

        return {}

    except Exception:
        return {}


def save_profile(profile):

    with open(
        PROFILE_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            profile,
            f,
            indent=4,
            ensure_ascii=False
        )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image(image):

    try:

        image_array = np.array(image)

        if len(image_array.shape) == 3:

            image_array = cv2.cvtColor(
                image_array,
                cv2.COLOR_RGB2BGR
            )

        hsv = cv2.cvtColor(
            image_array,
            cv2.COLOR_BGR2HSV
        )

        average_hsv = np.mean(
            hsv.reshape(-1, 3),
            axis=0
        )

        return {
            "hue": float(average_hsv[0]),
            "saturation": float(average_hsv[1]),
            "brightness": float(average_hsv[2]),
            "width": image.width,
            "height": image.height
        }

    except Exception:

        return {
            "hue": 0,
            "saturation": 0,
            "brightness": 0,
            "width": image.width,
            "height": image.height
        }


# ============================================================
# COLOR DETECTION
# ============================================================

def get_color_name(hsv):

    h = hsv["hue"]
    s = hsv["saturation"]
    v = hsv["brightness"]

    if v < 50:
        return "Black"

    if s < 35 and v > 200:
        return "White"

    if s < 45:
        return "Gray"

    if h < 10 or h >= 170:
        return "Red"

    if h < 25:
        return "Orange"

    if h < 35:
        return "Yellow"

    if h < 85:
        return "Green"

    if h < 100:
        return "Cyan"

    if h < 135:
        return "Blue"

    if h < 160:
        return "Purple"

    return "Pink"


# ============================================================
# CATEGORY SUGGESTION
# ============================================================

def suggest_category(filename):

    name = filename.lower()

    if "dress" in name:
        return "Dress"

    if "shirt" in name:
        return "Shirt"

    if "blouse" in name:
        return "Blouse"

    if (
        "tshirt" in name
        or "t-shirt" in name
        or "tee" in name
    ):
        return "T-Shirt"

    if "sweater" in name:
        return "Sweater"

    if "cardigan" in name:
        return "Cardigan"

    if "hoodie" in name:
        return "Hoodie"

    if "jacket" in name:
        return "Jacket"

    if "coat" in name:
        return "Coat"

    if "skirt" in name:
        return "Skirt"

    if "jean" in name:
        return "Jeans"

    if "pant" in name or "trouser" in name:
        return "Pants"

    if "short" in name:
        return "Shorts"

    if "shoe" in name:
        return "Shoes"

    if "bag" in name:
        return "Bag"

    if "accessory" in name:
        return "Accessory"

    return "Other"


# ============================================================
# SEARCH WARDROBE
# ============================================================

def search_wardrobe(
    wardrobe,
    query
):

    if not query:
        return wardrobe

    query = query.lower()

    results = []

    for item in wardrobe:

        searchable = " ".join([
            str(item.get("name", "")),
            str(item.get("category", "")),
            str(item.get("color", "")),
            str(item.get("notes", "")),
            " ".join(item.get("style_tags", []))
        ]).lower()

        if query in searchable:
            results.append(item)

    return results


# ============================================================
# SCORE ITEM
# ============================================================

def score_item(
    item,
    aesthetic
):

    rules = AESTHETIC_RULES.get(
        aesthetic,
        {}
    )

    score = 0

    item_color = item.get(
        "color",
        ""
    )

    item_category = item.get(
        "category",
        ""
    )

    notes = item.get(
        "notes",
        ""
    ).lower()

    style_tags = " ".join(
        item.get(
            "style_tags",
            []
        )
    ).lower()

    if item_color in rules.get(
        "colors",
        []
    ):
        score += 4

    if item_category in rules.get(
        "categories",
        []
    ):
        score += 3

    combined_text = (
        notes + " " + style_tags
    )

    for keyword in rules.get(
        "keywords",
        []
    ):

        if keyword.lower() in combined_text:
            score += 2

    return score


# ============================================================
# RECOMMEND ITEMS
# ============================================================

def recommend_items(
    wardrobe,
    aesthetic
):

    scored = []

    for item in wardrobe:

        score = score_item(
            item,
            aesthetic
        )

        if score > 0:

            scored.append(
                (
                    score,
                    item
                )
            )

    scored.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return [
        item
        for score, item
        in scored
    ]


# ============================================================
# CREATE OUTFIT
# ============================================================

def create_outfit(
    wardrobe,
    aesthetic
):

    recommendations = recommend_items(
        wardrobe,
        aesthetic
    )

    if not recommendations:
        return []

    preferred_order = [
        "Dress",
        "Top",
        "T-Shirt",
        "Blouse",
        "Shirt",
        "Sweater",
        "Cardigan",
        "Hoodie",
        "Jacket",
        "Coat",
        "Skirt",
        "Pants",
        "Jeans",
        "Shorts",
        "Shoes",
        "Bag",
        "Accessory"
    ]

    selected = []
    categories_used = set()

    for category in preferred_order:

        for item in recommendations:

            item_category = item.get(
                "category",
                "Other"
            )

            if (
                item_category == category
                and item_category not in categories_used
            ):

                selected.append(item)
                categories_used.add(item_category)

                break

    has_dress = any(
        item.get("category") == "Dress"
        for item in selected
    )

    if has_dress:

        selected = [
            item
            for item in selected
            if item.get("category")
            not in [
                "Pants",
                "Jeans",
                "Shorts",
                "Skirt"
            ]
        ]

    return selected


# ============================================================
# AI PROMPT
# ============================================================

def build_ai_prompt(
    user_message,
    wardrobe,
    profile
):

    wardrobe_text = []

    for item in wardrobe:

        wardrobe_text.append(
            f"- {item.get('name', 'Unnamed')} | "
            f"{item.get('category', '')} | "
            f"{item.get('color', '')} | "
            f"{item.get('notes', '')}"
        )

    wardrobe_context = "\n".join(
        wardrobe_text
    )

    if not wardrobe_context:
        wardrobe_context = "The wardrobe is currently empty."

    profile_context = json.dumps(
        profile,
        indent=2,
        ensure_ascii=False
    )

    return f"""
You are an AI wardrobe stylist.

The user wants practical outfit recommendations.

USER PROFILE:
{profile_context}

WARDROBE:
{wardrobe_context}

USER REQUEST:
{user_message}

Rules:
- Recommend clothing that actually exists in the user's wardrobe.
- Consider the user's preferred aesthetics.
- Consider occasion, comfort, colors and fit preferences.
- Do not judge the user's body.
- Do not rank attractiveness.
- Focus on clothing, styling and personal preferences.
- Give clear, useful outfit combinations.
"""


# ============================================================
# LOCAL AI
# ============================================================

def ask_local_ai(prompt):

    try:

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )

        if response.status_code != 200:

            return (
                "I couldn't connect to Ollama.\n\n"
                "Make sure Ollama is running."
            )

        data = response.json()

        return data.get(
            "response",
            "I couldn't generate a response."
        )

    except requests.exceptions.ConnectionError:

        return (
            "Ollama is not running.\n\n"
            "Start Ollama and make sure the "
            "llama3.2 model is installed."
        )

    except Exception as e:

        return f"AI error: {e}"


# ============================================================
# FASHN CATEGORY
# ============================================================

def get_fashn_category(category):

    tops = [
        "Top",
        "T-Shirt",
        "Shirt",
        "Blouse",
        "Sweater",
        "Cardigan",
        "Hoodie",
        "Jacket",
        "Coat"
    ]

    bottoms = [
        "Skirt",
        "Pants",
        "Jeans",
        "Shorts"
    ]

    if category in tops:
        return "tops"

    if category in bottoms:
        return "bottoms"

    if category == "Dress":
        return "one-pieces"

    return None


# ============================================================
# EXTRACT IMAGE FROM FASHN RESULT
# ============================================================

def extract_image_from_result(result):

    if result is None:
        return None

    # PIL image
    if isinstance(result, Image.Image):
        return result

    # Lists / tuples
    if isinstance(result, (list, tuple)):

        for item in result:

            extracted = extract_image_from_result(
                item
            )

            if extracted is not None:
                return extracted

        return None

    # Dictionaries
    if isinstance(result, dict):

        for key in [
            "image",
            "images",
            "output",
            "path",
            "url",
            "value"
        ]:

            if key in result:

                extracted = extract_image_from_result(
                    result[key]
                )

                if extracted is not None:
                    return extracted

        return None

    # String
    if isinstance(result, str):

        return result

    # FileData-like object
    if hasattr(result, "path"):

        path = getattr(
            result,
            "path",
            None
        )

        if path:
            return path

    if hasattr(result, "url"):

        url = getattr(
            result,
            "url",
            None
        )

        if url:
            return url

    return result


# ============================================================
# DISPLAY FASHN OUTPUT
# ============================================================

def display_tryon_output(output):

    if output is None:
        st.error("FASHN returned no output.")
        return

    # --------------------------------------------------------
    # PIL IMAGE
    # --------------------------------------------------------

    if isinstance(output, Image.Image):

        st.image(
            output,
            caption="AI Try-On Result",
            use_container_width=True
        )

        return

    # --------------------------------------------------------
    # STRING
    # --------------------------------------------------------

    if isinstance(output, str):

        # URL
        if (
            output.startswith("http://")
            or output.startswith("https://")
        ):

            try:

                response = requests.get(
                    output,
                    timeout=60
                )

                response.raise_for_status()

                result_image = Image.open(
                    BytesIO(response.content)
                ).convert("RGB")

                st.image(
                    result_image,
                    caption="AI Try-On Result",
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Could not download the generated image: {e}"
                )

            return

        # Local path
        if os.path.exists(output):

            try:

                result_image = Image.open(
                    output
                ).convert("RGB")

                st.image(
                    result_image,
                    caption="AI Try-On Result",
                    use_container_width=True
                )

            except Exception as e:

                st.error(
                    f"Could not open the generated image: {e}"
                )

            return

        st.write(output)
        return

    # --------------------------------------------------------
    # OTHER RESULT TYPES
    # --------------------------------------------------------

    extracted = extract_image_from_result(
        output
    )

    if extracted is not output:

        display_tryon_output(
            extracted
        )

    else:

        st.write(output)


# ============================================================
# GENERATE VIRTUAL TRY-ON
# ============================================================

def generate_virtual_tryon(
    human_image_path,
    garment_image_path,
    garment_category,
    garment_description
):

    # --------------------------------------------------------
    # CHECK CLIENT
    # --------------------------------------------------------

    if Client is None or handle_file is None:

        return (
            None,
            (
                "The Hugging Face Gradio client is not installed.\n\n"
                "Run this in Command Prompt:\n\n"
                "py -3.14 -m pip install -U gradio_client"
            )
        )

    # --------------------------------------------------------
    # CHECK FILES
    # --------------------------------------------------------

    if not os.path.exists(human_image_path):

        return (
            None,
            "The saved user photo could not be found."
        )

    if not os.path.exists(garment_image_path):

        return (
            None,
            "The selected wardrobe image could not be found."
        )

    # --------------------------------------------------------
    # CHECK CATEGORY
    # --------------------------------------------------------

    fashn_category = get_fashn_category(
        garment_category
    )

    if fashn_category is None:

        return (
            None,
            (
                f"FASHN VTON does not support "
                f"'{garment_category}'.\n\n"
                "Supported categories are:\n"
                "• tops\n"
                "• bottoms\n"
                "• one-pieces"
            )
        )

    # --------------------------------------------------------
    # GET HF TOKEN
    # --------------------------------------------------------

    hf_token = get_huggingface_token()

    if not hf_token:

        return (
            None,
            (
                "Hugging Face authentication is not configured.\n\n"
                "Create this file:\n\n"
                ".streamlit/secrets.toml\n\n"
                "and add:\n\n"
                'HF_TOKEN = "hf_your_token_here"\n\n'
                "Then restart Streamlit."
            )
        )

    # --------------------------------------------------------
    # CONNECT TO FASHN
    # --------------------------------------------------------

    try:

        client = Client(
            FASHN_SPACE,
            token=hf_token
        )

    except Exception as e:

        return (
            None,
            (
                "Could not connect to the FASHN Hugging Face Space.\n\n"
                f"{type(e).__name__}: {e}\n\n"
                "Check that your Hugging Face token is valid."
            )
        )

    # --------------------------------------------------------
    # CALL CURRENT FASHN API
    # --------------------------------------------------------

    try:

        result = client.predict(
            person_image=handle_file(
                human_image_path
            ),
            garment_image=handle_file(
                garment_image_path
            ),
            category=fashn_category,
            garment_photo_type="flat-lay",
            num_timesteps=30,
            guidance_scale=1.5,
            seed=42,
            segmentation_free=True,
            api_name="/try_on"
        )

        output = extract_image_from_result(
            result
        )

        if output is None:

            return (
                None,
                "FASHN returned no image."
            )

        return output, None

    # --------------------------------------------------------
    # QUOTA ERROR
    # --------------------------------------------------------

    except Exception as e:

        error_text = str(e)
        error_lower = error_text.lower()

        if (
            "zerogpu" in error_lower
            or "quota" in error_lower
            or "exceeded" in error_lower
        ):

            return (
                None,
                (
                    "Your Hugging Face ZeroGPU quota has been "
                    "exhausted.\n\n"
                    "This is a Hugging Face usage-limit issue, "
                    "not a problem with your wardrobe code.\n\n"
                    "Your token is being used for authentication, "
                    "but your account still has a daily GPU quota.\n\n"
                    "Try again after the quota resets."
                )
            )

        # ----------------------------------------------------
        # API ERROR
        # ----------------------------------------------------

        if (
            "api_name" in error_lower
            or "try_on" in error_lower
            or "not found" in error_lower
            or "endpoint" in error_lower
        ):

            return (
                None,
                (
                    "The FASHN API endpoint appears to have "
                    "changed.\n\n"
                    f"{type(e).__name__}: {e}\n\n"
                    "The app is currently using the official "
                    "FASHN /try_on endpoint."
                )
            )

        # ----------------------------------------------------
        # GENERAL ERROR
        # ----------------------------------------------------

        return (
            None,
            (
                "FASHN VTON virtual try-on failed.\n\n"
                f"{type(e).__name__}: {e}"
            )
        )


# ============================================================
# SESSION STATE
# ============================================================

if "wardrobe" not in st.session_state:

    st.session_state.wardrobe = load_wardrobe()


if "profile" not in st.session_state:

    st.session_state.profile = load_profile()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "👗 AI Wardrobe Stylist"
)

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "👚 My Wardrobe",
        "📸 Add Clothes",
        "📝 Style Quiz",
        "✨ Outfit Generator",
        "🤖 AI Stylist",
        "👤 Virtual Try-On"
    ]
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.title(
        "👗 AI Wardrobe Stylist"
    )

    st.subheader(
        "Your personal digital wardrobe"
    )

    st.write(
        """
        Organize your clothes, discover outfits,
        explore aesthetics and get AI-powered
        styling recommendations.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "👚 Wardrobe Items",
            len(st.session_state.wardrobe)
        )

    with col2:

        st.metric(
            "🎨 Aesthetics",
            len(AESTHETICS)
        )

    with col3:

        st.metric(
            "🤖 AI Stylist",
            "Ready"
        )

    st.divider()

    st.subheader(
        "✨ Features"
    )

    st.markdown(
        """
        - 👚 **Digital Wardrobe** — Keep track of your clothes.
        - 📸 **Add Clothes** — Upload photos of clothing.
        - 📝 **Style Quiz** — Tell the app about your style.
        - ✨ **Outfit Generator** — Create outfits from your wardrobe.
        - 🤖 **AI Stylist** — Ask for personalized recommendations.
        - 👤 **Virtual Try-On** — Preview wardrobe pieces using FASHN VTON AI.
        """
    )


# ============================================================
# MY WARDROBE
# ============================================================

elif page == "👚 My Wardrobe":

    st.header(
        "👚 My Wardrobe"
    )

    wardrobe = st.session_state.wardrobe

    search = st.text_input(
        "🔎 Search your wardrobe"
    )

    filtered = search_wardrobe(
        wardrobe,
        search
    )

    if not filtered:

        st.info(
            "No clothing items found."
        )

    else:

        columns = st.columns(4)

        for index, item in enumerate(filtered):

            with columns[index % 4]:

                image_path = item.get(
                    "image",
                    ""
                )

                if os.path.exists(image_path):

                    st.image(
                        image_path,
                        use_container_width=True
                    )

                st.markdown(
                    f"### {item.get('name', 'Unnamed')}"
                )

                st.write(
                    f"**Category:** "
                    f"{item.get('category', '')}"
                )

                st.write(
                    f"**Color:** "
                    f"{item.get('color', '')}"
                )

                if item.get("notes"):

                    st.caption(
                        item.get("notes")
                    )

                if st.button(
                    "🗑️ Delete",
                    key=f"delete_{item.get('id', index)}"
                ):

                    try:

                        if os.path.exists(
                            image_path
                        ):

                            os.remove(
                                image_path
                            )

                    except OSError:
                        pass

                    st.session_state.wardrobe = [
                        x
                        for x in wardrobe
                        if x.get("id")
                        != item.get("id")
                    ]

                    save_wardrobe(
                        st.session_state.wardrobe
                    )

                    st.rerun()


# ============================================================
# ADD CLOTHES
# ============================================================

elif page == "📸 Add Clothes":

    st.header(
        "📸 Add Clothes"
    )

    uploaded_files = st.file_uploader(
        "Upload photos of your clothes",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        accept_multiple_files=True
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            st.divider()

            try:

                image = Image.open(
                    uploaded_file
                ).convert("RGB")

                st.image(
                    image,
                    width=300
                )

                analysis = analyze_image(
                    image
                )

                detected_color = get_color_name(
                    analysis
                )

                suggested_cat = suggest_category(
                    uploaded_file.name
                )

                default_name = os.path.splitext(
                    uploaded_file.name
                )[0]

                name = st.text_input(
                    "Item name",
                    value=default_name,
                    key=f"name_{uploaded_file.name}"
                )

                category = st.selectbox(
                    "Category",
                    CATEGORIES,
                    index=(
                        CATEGORIES.index(
                            suggested_cat
                        )
                        if suggested_cat in CATEGORIES
                        else len(CATEGORIES) - 1
                    ),
                    key=f"category_{uploaded_file.name}"
                )

                color_options = [
                    "Black",
                    "White",
                    "Gray",
                    "Red",
                    "Orange",
                    "Yellow",
                    "Green",
                    "Cyan",
                    "Blue",
                    "Purple",
                    "Pink",
                    "Brown",
                    "Beige",
                    "Navy"
                ]

                color = st.selectbox(
                    "Color",
                    color_options,
                    index=(
                        color_options.index(
                            detected_color
                        )
                        if detected_color in color_options
                        else 0
                    ),
                    key=f"color_{uploaded_file.name}"
                )

                notes = st.text_area(
                    "Notes",
                    placeholder=(
                        "Example: floral, oversized, "
                        "lace, vintage..."
                    ),
                    key=f"notes_{uploaded_file.name}"
                )

                style_tags = st.multiselect(
                    "Style tags",
                    AESTHETICS,
                    key=f"tags_{uploaded_file.name}"
                )

                if st.button(
                    "➕ Add to Wardrobe",
                    key=f"add_{uploaded_file.name}"
                ):

                    file_id = str(
                        uuid.uuid4()
                    )

                    extension = os.path.splitext(
                        uploaded_file.name
                    )[1].lower()

                    if extension not in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                        ".webp"
                    ]:

                        extension = ".jpg"

                    image_path = os.path.join(
                        IMAGE_DIR,
                        file_id + extension
                    )

                    image.save(
                        image_path
                    )

                    new_item = {
                        "id": file_id,
                        "name": name.strip() or default_name,
                        "category": category,
                        "color": color,
                        "notes": notes,
                        "style_tags": style_tags,
                        "image": image_path,
                        "created_at": datetime.now().isoformat()
                    }

                    st.session_state.wardrobe.append(
                        new_item
                    )

                    save_wardrobe(
                        st.session_state.wardrobe
                    )

                    st.success(
                        f"Added {name} to your wardrobe!"
                    )

            except Exception as e:

                st.error(
                    f"Could not process image: {e}"
                )


# ============================================================
# STYLE QUIZ
# ============================================================

elif page == "📝 Style Quiz":

    st.header(
        "📝 Style Quiz"
    )

    st.write(
        "Tell me a little about your clothing preferences."
    )

    favorite_colors = st.multiselect(
        "🎨 What colors do you enjoy wearing?",
        [
            "Black",
            "White",
            "Gray",
            "Brown",
            "Beige",
            "Red",
            "Orange",
            "Yellow",
            "Green",
            "Cyan",
            "Blue",
            "Purple",
            "Pink"
        ]
    )

    preferred_fit = st.selectbox(
        "👚 What fit do you usually prefer?",
        [
            "Relaxed",
            "Regular",
            "Oversized",
            "Fitted",
            "It depends on the outfit"
        ]
    )

    comfort = st.multiselect(
        "🧸 What matters most to you?",
        [
            "Comfort",
            "Mobility",
            "Soft fabrics",
            "Easy layering",
            "Weather appropriate clothing",
            "Style"
        ]
    )

    occasions = st.multiselect(
        "📅 What do you usually dress for?",
        [
            "School",
            "College",
            "Work",
            "Casual outings",
            "Parties",
            "Travel",
            "Formal events",
            "At home"
        ]
    )

    favorite_aesthetics = st.multiselect(
        "✨ Which aesthetics do you like?",
        AESTHETICS
    )

    dislikes = st.text_area(
        "🚫 Anything you dislike wearing?",
        placeholder=(
            "Example: uncomfortable fabrics, "
            "certain colors, certain clothing types..."
        )
    )

    if st.button(
        "💾 Save My Style Profile"
    ):

        profile = {
            "favorite_colors": favorite_colors,
            "preferred_fit": preferred_fit,
            "comfort_priorities": comfort,
            "occasions": occasions,
            "favorite_aesthetics": favorite_aesthetics,
            "dislikes": dislikes
        }

        st.session_state.profile = profile

        save_profile(
            profile
        )

        st.success(
            "Your style profile has been saved!"
        )


# ============================================================
# OUTFIT GENERATOR
# ============================================================

elif page == "✨ Outfit Generator":

    st.header(
        "✨ Outfit Generator"
    )

    wardrobe = st.session_state.wardrobe

    if not wardrobe:

        st.warning(
            "Your wardrobe is empty. "
            "Add some clothes first."
        )

    else:

        aesthetic = st.selectbox(
            "Choose an aesthetic",
            AESTHETICS
        )

        if st.button(
            "✨ Generate Outfit"
        ):

            outfit = create_outfit(
                wardrobe,
                aesthetic
            )

            if not outfit:

                st.warning(
                    "I couldn't find enough matching "
                    "items for this aesthetic."
                )

            else:

                st.success(
                    f"Outfit generated for {aesthetic}!"
                )

                columns = st.columns(
                    min(
                        4,
                        len(outfit)
                    )
                )

                for index, item in enumerate(outfit):

                    with columns[index % len(columns)]:

                        image_path = item.get(
                            "image",
                            ""
                        )

                        if os.path.exists(
                            image_path
                        ):

                            st.image(
                                image_path,
                                use_container_width=True
                            )

                        st.subheader(
                            item.get(
                                "name",
                                "Unnamed"
                            )
                        )

                        st.caption(
                            f"{item.get('category', '')} • "
                            f"{item.get('color', '')}"
                        )


# ============================================================
# AI STYLIST
# ============================================================

elif page == "🤖 AI Stylist":

    st.header(
        "🤖 AI Stylist"
    )

    st.write(
        """
        Ask your AI stylist for outfit ideas using
        the clothes already in your wardrobe.
        """
    )

    st.info(
        "This feature uses Ollama running locally "
        "on your computer."
    )

    st.subheader(
        "💡 Example prompts"
    )

    st.write(
        """
        - What should I wear to college tomorrow?
        - Create a Dark Academia outfit.
        - Give me a casual outfit using my wardrobe.
        - What can I wear for a party?
        - Make an outfit using my favorite colors.
        """
    )

    user_message = st.chat_input(
        "Ask your AI stylist..."
    )

    if user_message:

        st.chat_message(
            "user"
        ).write(
            user_message
        )

        prompt = build_ai_prompt(
            user_message,
            st.session_state.wardrobe,
            st.session_state.profile
        )

        with st.spinner(
            "Stylist is thinking..."
        ):

            response = ask_local_ai(
                prompt
            )

        st.chat_message(
            "assistant"
        ).write(
            response
        )


# ============================================================
# VIRTUAL TRY-ON
# ============================================================

elif page == "👤 Virtual Try-On":

    st.header(
        "👤 Virtual Try-On"
    )

    st.write(
        """
        Preview a clothing item from your wardrobe
        using FASHN VTON 1.5.
        """
    )

    st.info(
        "💡 Your saved photo stays on this computer "
        "inside the wardrobe_data folder. "
        "The try-on request is sent to the FASHN Hugging Face Space."
    )

    # --------------------------------------------------------
    # HF STATUS
    # --------------------------------------------------------

    if Client is None:

        st.warning(
            "Hugging Face support is not installed."
        )

        st.code(
            "py -3.14 -m pip install -U gradio_client",
            language="text"
        )

    else:

        hf_token = get_huggingface_token()

        if hf_token:

            st.success(
                "✅ Hugging Face authentication is configured."
            )

        else:

            st.warning(
                "⚠️ Hugging Face authentication is not configured."
            )

            st.code(
                """
.streamlit/secrets.toml

HF_TOKEN = "hf_your_token_here"
                """,
                language="toml"
            )

    # --------------------------------------------------------
    # USER PHOTO
    # --------------------------------------------------------

    st.subheader(
        "📸 Your Photo"
    )

    uploaded_photo = st.file_uploader(
        "Upload a photo of yourself",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ],
        key="user_photo_upload"
    )

    if uploaded_photo:

        try:

            user_image = Image.open(
                uploaded_photo
            ).convert("RGB")

            st.image(
                user_image,
                caption="Photo preview",
                width=350
            )

            if st.button(
                "💾 Save My Photo",
                key="save_user_photo"
            ):

                user_image.save(
                    USER_PHOTO_FILE,
                    "JPEG",
                    quality=95
                )

                st.success(
                    "Your photo has been saved!"
                )

                st.rerun()

        except Exception as e:

            st.error(
                f"Could not process the photo: {e}"
            )

    # --------------------------------------------------------
    # SAVED PHOTO
    # --------------------------------------------------------

    saved_photo_exists = os.path.exists(
        USER_PHOTO_FILE
    )

    saved_image = None

    if saved_photo_exists:

        st.divider()

        st.subheader(
            "🪞 Your Saved Photo"
        )

        try:

            saved_image = Image.open(
                USER_PHOTO_FILE
            ).convert("RGB")

            st.image(
                saved_image,
                width=350
            )

        except Exception as e:

            st.error(
                f"Could not open your saved photo: {e}"
            )

            saved_photo_exists = False

    # --------------------------------------------------------
    # TRY-ON
    # --------------------------------------------------------

    if saved_photo_exists:

        st.divider()

        st.subheader(
            "👗 Choose clothes to preview"
        )

        wardrobe = st.session_state.wardrobe

        if not wardrobe:

            st.warning(
                "Your wardrobe is empty. "
                "Add some clothes first!"
            )

        else:

            valid_items = [
                item
                for item in wardrobe
                if os.path.exists(
                    item.get("image", "")
                )
            ]

            if not valid_items:

                st.warning(
                    "Your wardrobe does not contain "
                    "any usable clothing images."
                )

            else:

                # ------------------------------------------------
                # Use item ID internally.
                # This allows two clothes to have the same name.
                # ------------------------------------------------

                selected_item_id = st.selectbox(
                    "Choose a clothing item",
                    options=[
                        item.get("id")
                        for item in valid_items
                    ],
                    format_func=lambda item_id: next(
                        (
                            item.get("name", "Unnamed item")
                            for item in valid_items
                            if item.get("id") == item_id
                        ),
                        "Unnamed item"
                    ),
                    key="try_on_item"
                )

                selected_item = next(
                    (
                        item
                        for item in valid_items
                        if item.get("id") == selected_item_id
                    ),
                    None
                )

                if selected_item:

                    col1, col2 = st.columns(2)

                    # ------------------------------------------------
                    # PERSON
                    # ------------------------------------------------

                    with col1:

                        st.subheader(
                            "🧍 You"
                        )

                        st.image(
                            saved_image,
                            use_container_width=True
                        )

                    # ------------------------------------------------
                    # GARMENT
                    # ------------------------------------------------

                    with col2:

                        st.subheader(
                            "👚 Selected item"
                        )

                        item_image_path = selected_item.get(
                            "image",
                            ""
                        )

                        st.image(
                            item_image_path,
                            use_container_width=True
                        )

                        st.write(
                            f"**{selected_item.get('name', 'Unnamed')}**"
                        )

                        st.caption(
                            f"{selected_item.get('category', '')} • "
                            f"{selected_item.get('color', '')}"
                        )

                    st.divider()

                    st.subheader(
                        "✨ Try-On Preview"
                    )

                    category = selected_item.get(
                        "category",
                        "Other"
                    )

                    fashn_category = get_fashn_category(
                        category
                    )

                    notes = selected_item.get(
                        "notes",
                        ""
                    )

                    color = selected_item.get(
                        "color",
                        ""
                    )

                    garment_description = (
                        f"{color} {category}"
                    )

                    if notes:

                        garment_description += (
                            f", {notes}"
                        )

                    if fashn_category:

                        st.write(
                            f"**FASHN category:** "
                            f"`{fashn_category}`"
                        )

                        st.write(
                            f"**Garment description:** "
                            f"{garment_description}"
                        )

                    else:

                        st.warning(
                            "This wardrobe category is not "
                            "supported by FASHN VTON 1.5."
                        )

                        st.markdown(
                            """
                            FASHN VTON 1.5 supports:

                            - 👚 **Tops**
                            - 👖 **Bottoms**
                            - 👗 **One-pieces**
                            """
                        )

                    if category in [
                        "Shoes",
                        "Bag",
                        "Accessory",
                        "Other"
                    ]:

                        st.warning(
                            "Shoes, bags, accessories and other "
                            "non-supported items cannot be used "
                            "for this virtual try-on."
                        )

                    # ------------------------------------------------
                    # GENERATE BUTTON
                    # ------------------------------------------------

                    generate_button = st.button(
                        "✨ Generate AI Try-On",
                        type="primary",
                        key="generate_tryon"
                    )

                    if generate_button:

                        if Client is None:

                            st.error(
                                "Please install gradio_client first."
                            )

                        elif not get_huggingface_token():

                            st.error(
                                "Hugging Face authentication is not configured."
                            )

                            st.info(
                                """
                                Create:

                                .streamlit/secrets.toml

                                and put:

                                HF_TOKEN = "hf_your_token_here"

                                Then restart Streamlit.
                                """
                            )

                        elif fashn_category is None:

                            st.error(
                                f"'{category}' cannot be used "
                                "with FASHN VTON 1.5."
                            )

                        else:

                            with st.spinner(
                                """
                                FASHN VTON 1.5 is generating
                                your try-on preview...

                                This can take a little while.
                                """
                            ):

                                output, error = generate_virtual_tryon(
                                    USER_PHOTO_FILE,
                                    selected_item.get("image", ""),
                                    category,
                                    garment_description
                                )

                            if error:

                                st.error(
                                    error
                                )

                            elif output is not None:

                                st.success(
                                    "🎉 Try-on generated successfully!"
                                )

                                display_tryon_output(
                                    output
                                )

                    st.caption(
                        """
                        FASHN VTON 1.5 is provided through the
                        FASHN AI Hugging Face Space. Review the
                        model and third-party component licenses
                        before commercial deployment.
                        """
                    )

        # --------------------------------------------------------
        # DELETE PHOTO
        # --------------------------------------------------------

        st.divider()

        if st.button(
            "🗑️ Remove My Saved Photo",
            key="delete_user_photo"
        ):

            try:

                os.remove(
                    USER_PHOTO_FILE
                )

                st.success(
                    "Your saved photo was removed."
                )

                st.rerun()

            except OSError as e:

                st.error(
                    f"Could not remove the photo: {e}"
                )

    else:

        st.info(
            "Upload and save a photo above to start."
        )
