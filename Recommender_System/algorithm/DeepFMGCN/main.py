if __name__ == '__main__':
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam

    from Recommender_System.data import kg_loader, data_process
    from Recommender_System.algorithm.DeepFMGCN.model import DeepFM_model
    from Recommender_System.algorithm.DeepFMGCN.train import train

    # n_user, n_item, n_entity, n_relation, train_data, test_data, kg, topk_data = data_process.pack_kg(kg_loader.ml1m_kg20k,
    #                                                                                                   keep_all_head=False,
    #                                                                                                   negative_sample_threshold=4)
    # model_rs, model_kge = DeepFM_model(n_user, n_item, n_entity, n_relation, dim=8, L=1, H=1, l2=1e-6)
    # train(model_rs, model_kge, train_data, test_data, kg, topk_data, kge_interval=3,
    #       optimizer_rs=Adam(0.02), optimizer_kge=Adam(0.01), epochs=2, batch=4096)


    # The Philippine Jurisprudence
    n_user, n_item, n_entity, n_relation, train_data, test_data, kg, topk_data = data_process.pack_kg(kg_loader.phil_juris,
                                                                                                      keep_all_head=False,
                                                                                                      negative_sample_threshold=3)
    model_rs, model_kge = DeepFM_model(n_user, n_item, n_entity, n_relation, dim=8, L=1, H=1, l2=1e-6)
    train(model_rs, model_kge, train_data, test_data, kg, topk_data, kge_interval=2,
          optimizer_rs=Adam(0.02), optimizer_kge=Adam(0.01), epochs=3, batch=4096)

    tf.saved_model.save(
        model_rs,
        "saved_model",
        signatures=None,
    )
    # n_user, n_item, n_entity, n_relation, train_data, test_data, kg, topk_data = data_process.pack_kg(kg_loader.lastfm_kg15k, keep_all_head=False)
    # model_rs, model_kge = DeepFM_model(n_user, n_item, n_entity, n_relation, dim=4, L=5, H=1, l2=1e-6)
    # train(model_rs, model_kge, train_data, test_data, kg, topk_data, kge_interval=2,
    #       optimizer_rs=Adam(1e-3), optimizer_kge=Adam(2e-4), epochs=1, batch=256)

    # n_user, n_item, n_entity, n_relation, train_data, test_data, kg, topk_data = data_process.pack_kg(kg_loader.bx_kg20k, keep_all_head=False)
    # model_rs, model_kge = DeepFM_model(n_user, n_item, n_entity, n_relation, dim=8, L=1, H=1, l2=1e-6)
    # train(model_rs, model_kge, train_data, test_data, kg, topk_data, kge_interval=2,
    #       optimizer_rs=Adam(2e-4), optimizer_kge=Adam(2e-5), epochs=10, batch=32)
    #
    # user_id = 100
    #
    # movie_data = tf.constant([d[1] for d in test_data])
    # head_data = tf.constant([d[1] for d in test_data])
    # user = tf.constant([user_id for d in test_data])
    #
    #
    # predictions = model_rs.predict({"user_id": user, "item_id": movie_data, "head_id": head_data})
    # print(predictions)

    #
    # tfjs.converters.save_keras_model(model_rs, "")