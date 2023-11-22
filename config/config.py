import ml_collections


def config_train():
    config = ml_collections.ConfigDict()
    # !random parameter setting
    config.seed = 43

    # !distributed training setting
    config.gpus = 2
    config.world_size = 2
    config.backend = 'nccl'
    config.init_method = 'tcp://10.249.178.201:1234'
    config.syncbn = True

    # !transform setting
    config.face = 'scale'
    config.size = 32

    # !training parameter setting
    config.debug = False
    config.logint = 100
    config.modelperiod = 500
    config.valint = 500

    config.batch = 128
    config.batch_train = 32
    config.epochs = 1000
    config.net_s = 'WholeNet'
    config.net_t = 'SeqNet'
    config.net_t_path = "/mnt/8T/hou/multicard_teacher/weights/binclass/net-xception_traindb-ff-c23-720-140-140_face-scale_size-299_seed-22_note-test/bestval.pth"
    # config.traindb = ["dfdc-30-10-10"]
    config.traindb = ["ff-c23-720-140-140"]

    config.traindb2 = ["celebdf-500-50-40"]
    config.trainIndex = 0
    config.tagnote = 'test-101'

    # !dataset setting
    config.ffpp_faces_df_path = '/mnt/8T/hou/FFPP/df/output/FFPP_df.pkl'
    config.ffpp_faces_dir = '/mnt/8T/hou/FFPP/faces/output'
    config.dfdc_faces_df_path = '/mnt/8T/hou/dfdc_faces/faces_df.pkl'
    config.dfdc_faces_dir = '/mnt/8T/hou/dfdc_faces'
    config.celebdf_faces_df_path = '/mnt/8T/hou/celeb-df/faces/celeb_df.pkl'
    config.celebdf_faces_dir = '/mnt/8T/hou/celeb-df/faces'
    config.workers = 4

    # !optimizer setting
    config.lr = 1e-3
    config.patience = 10

    # !model loading setting
    config.models_dir = '/mnt/8T/hou/multicard_forTest/weights/binclass/'
    config.mode = 0
    config.index = 0

    # !log setting
    config.log_dir = '/mnt/8T/hou/multicard_forTest/runs/binclass/'

    return config


def config_test():
    config = ml_collections.ConfigDict()
    # !random parameter setting
    config.seed = 43

    # !distributed training setting
    config.gpus = 2
    config.world_size = 2
    config.backend = 'nccl'
    config.init_method = 'tcp://10.249.41.225:12345'
    config.syncbn = True

    # !transform setting
    config.face = 'scale'
    config.size = 299

    # !training parameter setting
    config.debug = False
    config.logint = 100
    config.modelperiod = 500
    config.valint = 500

    config.batch = 40
    config.batch_train = 32
    config.epochs = 26
    config.net_s = 'WholeNet'
    config.net_t = 'CNet'
    config.net_t_path = "/mnt/8T/hou/multicard_teacher/weights/binclass/net-xception_traindb-ff-c23-720-140-140_face-scale_size-299_seed-22_note-test/bestval.pth"
    # config.traindb = ["dfdc-30-10-10"]
    config.traindb1 = ["ff-c23-720-140-140"]

    config.traindb2 = ["celebdf-500-50-100"]
    config.trainIndex = 0
    config.tagnote = 'DF_1'

    # !dataset setting
    config.ffpp_faces_df_path = '/mnt/2T/FFPP/df/output/FFPP_df.pkl'
    config.ffpp_faces_dir = '/mnt/2T/FFPP/faces/output'
    config.dfdc_faces_df_path = '/mnt/8T/hou/dfdc_faces/faces_df.pkl'
    config.dfdc_faces_dir = '/mnt/8T/hou/dfdc_faces'
    config.celebdf_faces_df_path = '/media/ubuntu/celeb-df/faces/celeb_df.pkl'
    config.celebdf_faces_dir = '/media/ubuntu/celeb-df/faces'
    config.workers = 8

    # !optimizer setting
    config.lr = 1e-4
    config.patience = 10

    # !model loading setting
    config.models_dir = '/media/ubuntu/hou/multicard_CNET/weights/binclass/'
    config.mode = 1
    config.index = 20

    # !log setting
    config.log_dir = '/media/ubuntu/hou/multicard_CNET/runs/binclass/'

    return config


def config_3():
    config = ml_collections.ConfigDict()
    # !random parameter setting
    config.seed = 43

    # !distributed training setting
    config.gpus = 2
    config.world_size = 2
    config.backend = 'nccl'
    config.init_method = 'tcp://10.249.178.201:1234'
    config.syncbn = True

    # !transform setting
    config.face = 'scale'
    config.size_images = 299
    config.size_cuttings = 32

    # !training parameter setting
    config.debug = False
    config.logint = 100
    config.modelperiod = 500
    config.valint = 500

    config.batch = 32
    config.batch_train = 32
    config.epochs = 1000
    config.net_s = 'WholeNet'
    config.net_t = 'xception'
    config.net_t_path = "/mnt/8T/hou/multicard_teacher/weights/binclass/net-xception_traindb-ff-c23-720-140-140_face-scale_size-299_seed-22_note-test/bestval.pth"
    # config.traindb = ["dfdc-30-10-10"]

    config.traindb = ["ff-c23-720-140-140"]
    config.traindb2 = ["celebdf-500-50-40"]

    config.trainIndex = 0
    config.tagnote = 't1'

    # !dataset setting
    config.ffpp_faces_df_path = '/mnt/8T/hou/FFPP/df/output/FFPP_df.pkl'
    config.ffpp_faces_dir = '/mnt/8T/hou/FFPP/faces/output'
    config.dfdc_faces_df_path = '/mnt/8T/hou/dfdc_faces/faces_df.pkl'
    config.dfdc_faces_dir = '/mnt/8T/hou/dfdc_faces'
    config.celebdf_faces_df_path = '/mnt/8T/hou/celeb-df/faces/celeb_df.pkl'
    config.celebdf_faces_dir = '/mnt/8T/hou/celeb-df/faces'
    config.workers = 4

    # !optimizer setting
    config.lr = 1e-3
    config.patience = 10

    # !model loading setting
    config.models_dir = '/mnt/8T/hou/multicard_forTest/weights/binclass/'
    config.mode = 0
    config.index = 0

    # !log setting
    config.log_dir = '/mnt/8T/hou/multicard_forTest/runs/binclass/'

    return config
