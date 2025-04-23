from Bloom.BloomRXN.Qdrant.QdrantBase import QdrantBase


class RXNReference(QdrantBase):
    def __init__(self):
        super().__init__(
            collection_name="rxn_ec",
            memory_strategy="disk",
            label_alias="ec",
            embedding_dim=128,
            memmap_threshold=20000,
            delete_existing=False,
        )
