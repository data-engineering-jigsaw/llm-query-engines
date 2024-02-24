def persist_data(index, location = "./storage"):
    index.set_index_id("vector_index")
    index.storage_context.persist(location)
    storage_context = StorageContext.from_defaults(persist_dir="storage")
    return storage_context

# index = load_index_from_storage(storage_context, index_id="vector_index")