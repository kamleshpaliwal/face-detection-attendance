from pymilvus import DataType, MilvusClient

from .config import settings


class FaceVectors:
    def __init__(self):
        address = settings.milvus_address
        if "://" not in address:
            address = f"http://{address}"
        self.client = MilvusClient(uri=address)
        self.collection = settings.milvus_collection

    def initialize(self) -> None:
        if not self.client.has_collection(self.collection):
            schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)
            schema.add_field("student_id", DataType.INT64, is_primary=True)
            schema.add_field("embedding", DataType.FLOAT_VECTOR, dim=settings.face_dimension)
            index = self.client.prepare_index_params()
            index.add_index(
                field_name="embedding",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": 16, "efConstruction": 200},
            )
            self.client.create_collection(
                collection_name=self.collection, schema=schema, index_params=index
            )
        self.client.load_collection(self.collection)

    def put(self, student_id: int, embedding: list[float]) -> None:
        self.client.upsert(
            collection_name=self.collection,
            data=[{"student_id": student_id, "embedding": embedding}],
        )

    def nearest(self, embedding: list[float]) -> dict | None:
        result = self.client.search(
            collection_name=self.collection,
            data=[embedding],
            anns_field="embedding",
            limit=1,
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            output_fields=["student_id"],
        )
        hits = result[0] if result else []
        if not hits:
            return None
        hit = hits[0]
        return {"student_id": int(hit["entity"]["student_id"]), "score": float(hit["distance"])}


face_vectors = FaceVectors()
