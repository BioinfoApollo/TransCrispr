ModelParams_WT = {
    'train_batch_size':20,
    'train_epochs_num':100,
    'train_base_learning_rate':0.00002,
    'model_save_file':'./models/BestModel_WT.h5',
    'dropout_rate':0.4,
    'nuc_embedding_outputdim':66,
    'conv1d_filters_size':7,
    'conv1d_filters_num':512,
    'transformer_num_layers':4,
    'transformer_final_fn':198,
    'transformer_ffn_1stlayer':111,
    'dense1':176,
    'dense2':88,
    'dense3':22
}

ModelParams_ESP = {
    'train_batch_size':20,
    'train_epochs_num':100,
    'train_base_learning_rate':0.00002,
    'model_save_file':'./models/BestModel_ESP.h5',
    'dropout_rate':0.2,
    'nuc_embedding_outputdim':61,
    'conv1d_filters_size':6,
    'conv1d_filters_num':256,
    'transformer_num_layers':3,
    'transformer_final_fn':107,
    'transformer_ffn_1stlayer':156,
    'dense1':159,
    'dense2':72,
    'dense3':40
}

ModelParams_HF = {
    'train_batch_size':32,
    'train_epochs_num':100,
    'train_base_learning_rate':0.00002,
    'model_save_file':'./models/BestModel_HF.h5',
    'dropout_rate':0.4,
    'nuc_embedding_outputdim':95,
    'conv1d_filters_size':5,
    'conv1d_filters_num':256,
    'transformer_num_layers':5,
    'transformer_final_fn':194,
    'transformer_ffn_1stlayer':157,
    'dense1':140,
    'dense2':51,
    'dense3':21
}

ModelParams_xCas = {
    'train_batch_size':64,
    'train_epochs_num':100,
    'train_base_learning_rate':0.0001,
    'model_save_file':'./models/BestModel_xCas.h5',
    'dropout_rate':0.4,
    'nuc_embedding_outputdim':36,
    'conv1d_filters_size':8,
    'conv1d_filters_num':256,
    'transformer_num_layers':4,
    'transformer_final_fn':64,
    'transformer_ffn_1stlayer':176,
    'dense1':114,
    'dense2':88,
    'dense3':45
}

ModelParams_SniperCas = {
    'train_batch_size':128,
    'train_epochs_num':100,
    'train_base_learning_rate':0.0001,
    'model_save_file':'./models/BestModel_SniperCas.h5',
    'dropout_rate':0.3,
    'nuc_embedding_outputdim':77,
    'conv1d_filters_size':5,
    'conv1d_filters_num':512,
    'transformer_num_layers':4,
    'transformer_final_fn':80,
    'transformer_ffn_1stlayer':115,
    'dense1':163,
    'dense2':65,
    'dense3':42
}

ModelParams_SpCas9 = {
    'train_batch_size':128,
    'train_epochs_num':100,
    'train_base_learning_rate':0.00008,
    'model_save_file':'./models/BestModel_SpCas9.h5',
    'dropout_rate':0.4,
    'nuc_embedding_outputdim':40,
    'conv1d_filters_size':8,
    'conv1d_filters_num':256,
    'transformer_num_layers':3,
    'transformer_final_fn':99,
    'transformer_ffn_1stlayer':181,
    'dense1':162,
    'dense2':53,
    'dense3':38
}

ModelParams_HypaCas9 = {
    'train_batch_size':64,
    'train_epochs_num':100,
    'train_base_learning_rate':0.0001,
    'model_save_file':'./models/BestModel_HypaCas9.h5',
    'dropout_rate':0.4,
    'nuc_embedding_outputdim':91,
    'conv1d_filters_size':6,
    'conv1d_filters_num':512,
    'transformer_num_layers':2,
    'transformer_final_fn':63,
    'transformer_ffn_1stlayer':143,
    'dense1':128,
    'dense2':64,
    'dense3':32
}

ParamsRanges = {
    'ModelParams_WT':
    {
        'train_base_learning_rate':[0.00001,0.0001],
        'dropout_rate':[0.1,0.5]
    },
    'ModelParams_ESP':
    {
        'dense1': [100, 200],
        'dense2': [50, 100],
        'dense3': [30, 50]
    },
    'ModelParams_HF':
    {
        'dense1': [100, 200],
        'dense2': [50, 100],
        'dense3': [10, 50]
    },
    'ModelParams_xCas':
    {
        'dense1': [100, 200],
        'dense2': [50, 100],
        'dense3': [10, 50]
    },
    'ModelParams_SniperCas':
    {
        'dense1': [100, 200],
        'dense2': [50, 100],
        'dense3': [10, 50]
    },
    'ModelParams_SpCas9':
    {
        'dense1': [100, 200],
        'dense2': [50, 100],
        'dense3': [10, 50]
    },
    'ModelParams_HypaCas9':
    {
        'dense1': [100, 200],
        'dense2': [50, 100],
        'dense3': [10, 50]
    }
}

Params = {
    'ModelParams':ModelParams_HF,
    'ParamsRanges':ParamsRanges
    }