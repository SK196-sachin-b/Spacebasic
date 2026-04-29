import os
import json
import boto3
from dotenv import load_dotenv

load_dotenv(override=True)
print("📄 Loaded environment variables")

class EmbeddingService:
    def __init__(self):
        aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        aws_session_token = os.getenv("AWS_SESSION_TOKEN")
        aws_region = os.getenv("AWS_REGION", "us-east-1")

        # 🔧 DEBUG: Print credentials (masked for safety)
        print("\n🔧 DEBUG: AWS Credentials Loaded")
        print(f"AWS_REGION: {aws_region}")
        print(f"AWS_ACCESS_KEY_ID: {aws_access_key[:4]}****{aws_access_key[-4:] if aws_access_key else None}")
        print(f"AWS_SECRET_ACCESS_KEY: {aws_secret_key[:4]}****{aws_secret_key[-4:] if aws_secret_key else None}")
        print(f"AWS_SESSION_TOKEN: {'Present' if aws_session_token else 'Not Present'}\n")

        self.bedrock_runtime = boto3.client(
            "bedrock-runtime",
            region_name=aws_region,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_session_token=aws_session_token,
        )

        self.embed_model_id = "amazon.titan-embed-text-v2:0"
    
    def embed_text(self, text):
        """Generate embedding for a single text"""
        try:
            print(f"   🔧 DEBUG: Embedding text of length {len(text)} chars")
            
            payload = json.dumps({"inputText": text})
            print(f"   🔧 DEBUG: Calling Bedrock with model {self.embed_model_id}")
            
            response = self.bedrock_runtime.invoke_model(
                modelId=self.embed_model_id, 
                body=payload
            )
            
            result = json.loads(response["body"].read())
            embedding = result["embedding"]
            
            print(f"   ✅ Embedding generated: {len(embedding)} dimensions")
            print(f"   🔧 DEBUG: Embedding type: {type(embedding)}")
            print(f"   🔧 DEBUG: First 3 values: {embedding[:3] if len(embedding) >= 3 else embedding}")
            
            return embedding
            
        except Exception as e:
            print(f"❌ Embedding generation failed: {e}")
            print(f"🔧 DEBUG: Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return None
    
    def embed_batch(self, texts):
        """Generate embeddings for multiple texts"""
        embeddings = []
        for i, text in enumerate(texts):
            print(f"Generating embedding {i+1}/{len(texts)}")
            embedding = self.embed_text(text)
            if embedding:
                embeddings.append(embedding)
            else:
                print(f"❌ Failed to generate embedding for text {i+1}")
        return embeddings

# Global embedding service instance
embedding_service = EmbeddingService()