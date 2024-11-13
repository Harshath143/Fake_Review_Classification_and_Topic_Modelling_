
# Fake Review Classification and Topic Modeling

## Overview

This project focuses on building machine learning and deep learning models to classify fake reviews and perform topic modeling on review text data. The model uses **BERT (Bidirectional Encoder Representations from Transformers)** for text classification and **Latent Dirichlet Allocation (LDA)** for topic modeling. The application is deployed as a **FastAPI** app, providing an API endpoint for classifying reviews and extracting topics.

### Features:
- **Fake Review Classification**: Classifies reviews as fake or real based on the content using a fine-tuned **BERT** model.
- **Topic Modeling**: Identifies topics from a collection of reviews using **LDA** (Latent Dirichlet Allocation).
- **FastAPI Backend**: A FastAPI application to handle incoming requests and serve predictions.
- **Dockerized**: Docker containerization for easy deployment and scaling.

---

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.7+**
- **Docker** (for containerizing the app)
- **Postman** (for testing the API endpoint)
- **Git** (for version control)

### Required Libraries

You will need the following Python libraries, which are listed in the `requirements.txt`:

- **fastapi**
- **uvicorn**
- **torch**
- **transformers**
- **pydantic**
- **nltk**
- **gensim**
- **sklearn**

These dependencies are required to run the app, the BERT model, and perform topic modeling.

---

## Installation

### 1. Clone the Repository

Start by cloning the repository:

```bash
git clone https://github.com/Harshath143/Fake_Review_Classification_and_Topic_Modelling_.git
cd your-repo-directory
```

### 2. Set Up Virtual Environment

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scriptsctivate
```

### 3. Install Dependencies

Install the necessary Python packages:

```bash
pip install -r requirements.txt
```

---

## Running the FastAPI App Locally

1. **Start FastAPI Server**: You can start the FastAPI application using **Uvicorn**:

   ```bash
   uvicorn app:app --reload
   ```

   This will run the FastAPI app locally at `http://127.10.1.1:8000`.

2. **API Documentation**: FastAPI automatically generates an interactive API documentation. You can access it by visiting:

   ```
   http://127.10.1.1:8000/docs
   ```

3. **Testing the API with Postman**:
   - Use **Postman** to send POST requests to the `/predict` endpoint with the review text to classify as fake or real.
   - The JSON payload should look like this:

   ```json
   {
     "text": "The product quality is excellent. I would highly recommend it!"
   }
   ```

   Example response:
   ```json
   {
     "label": "Real",
     "class_id": 1
   }
   ```

---

## Dockerizing the Application

1. **Build the Docker Image**: To build the Docker image, run the following command in the project directory (where the `Dockerfile` is located):

   ```bash
   docker build -t fastapi-fake-review .
   ```

2. **Run the Docker Container**: After building the image, run the following command to start the container:

   ```bash
   docker run -d -p 8000:8000 fastapi-fake-review
   ```

   - This will run the FastAPI app inside a container and map port `8000` of the container to port `8000` on your machine.
   - You can now access the app at `http://localhost:8000`.

3. **Use Docker Compose** (Optional): For easier multi-container management, you can use Docker Compose. Here's a sample `docker-compose.yml`:

   ```yaml
   version: '3.11'

   services:
     app:
       build: .
       ports:
         - "8000:8000"
       volumes:
         - .:/app
       environment:
         - PYTHONUNBUFFERED=1
   ```

   To run the app with Docker Compose, use:

   ```bash
   docker-compose up --build
   ```

---

## File Structure

```
/project-directory
├── app.py                  # FastAPI app for serving predictions
├── Dockerfile              # Dockerfile to containerize the app
├── requirements.txt        # Python dependencies for the project
├── saved_model/            # Directory containing the trained BERT model
├── /topic_modeling         # Directory containing scripts for topic modeling
├── /tests                  # Directory for test cases (if applicable)
└── docker-compose.yml      # (Optional) Docker Compose configuration for multi-container apps
```

---

## How the Model Works

### Fake Review Classification:
- **BERT Model**: The model used for fake review classification is based on **BERT for sequence classification**. It is trained to classify text into two classes: fake or real. This was trained with a combination of traditional machine learning algorithms (like **Multinomial Naive Bayes**, **Random Forest**, **Decision Trees**, **Logistic Regression**, and **KNN**) as well as deep learning models (such as **RNN**, **LSTM**, and **BiLSTM**) to enhance classification accuracy.
- **FastAPI**: The FastAPI app exposes a POST endpoint `/predict` which takes a JSON object containing a review and classifies it as fake or real.

### Topic Modeling:
- **LDA (Latent Dirichlet Allocation)**: The project also includes a **topic modeling** component using **LDA**. This helps identify the main topics from a collection of review texts. The LDA model extracts latent topics based on review content, providing insight into customer concerns or product features.

---

## Example Request

To classify a review as fake or real, send the following POST request to `http://127.0.0.1:8000/predict`:

### Request:

```json
{
  "text": "This product is terrible. It broke after one use. Do not buy!"
}
```

### Response:

```json
{
  "label": "Fake",
  "class_id": 0
}
```

---

## Troubleshooting

- **CUDA Errors**: If you're running on a machine with a GPU, you may encounter errors related to CUDA. Ensure that your CUDA drivers and PyTorch are correctly installed. You can disable oneDNN optimizations to avoid these errors by setting the environment variable:

  ```bash
  set TF_ENABLE_ONEDNN_OPTS=0
  ```

- **Model Loading Issues**: If the model is not loading correctly, make sure the path to the model directory (`saved_model`) is correct and contains the necessary files (`pytorch_model.bin`, `config.json`, and `vocab.txt`).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
)

