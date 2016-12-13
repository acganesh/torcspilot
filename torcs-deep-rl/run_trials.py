from main import main

config = {'train': 1,
          'network': 'FCNet_terminating',
          'experiment_name': 'aalborg',
          'EXPERIMENTS_PATH': './experiments/'}

limit = 1000
for i in xrange(1, 1000):
    network = 'FCNet_terminating' + str(i)
    config['network'] = network
    main(config)
