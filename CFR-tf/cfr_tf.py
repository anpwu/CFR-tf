import random
import numpy as np
import pandas as pd 
import tensorflow as tf
from module import Net

class CausalDB(object):
    def __init__(self, df):
        x_cols=[c for c in df.columns if 'x' in c]
        self.x = df[x_cols].values
        self.t = df['t'].values.reshape(-1,1)
        self.yf = df['yf'].values.reshape(-1,1)
        self.ycf = df['ycf'].values.reshape(-1,1)
        self.hat_y0 = df['hat_y0'].values.reshape(-1,1)
        self.hat_y1 = df['hat_y1'].values.reshape(-1,1)
        self.mu0 = df['mu0'].values.reshape(-1,1)
        self.mu1 = df['mu1'].values.reshape(-1,1)

def get_DataFrame(exp=0, path='data/IHDP_df.npz'):
    dataloader = np.load(path,allow_pickle=True,encoding='latin1')
    datas = dataloader['data']

    train_df = datas[exp]['train']
    valid_df = datas[exp]['valid']
    test_df = datas[exp]['test']
    data_df = pd.concat([train_df,valid_df,test_df])

    train_data = CausalDB(train_df)
    valid_data = CausalDB(valid_df)
    test_data = CausalDB(test_df)
    data_data = CausalDB(data_df)

    return train_data, valid_data, test_data, data_data

def simplex_project(x,k):
    """ Projects a vector x onto the k-simplex """
    d = x.shape[0]
    mu = np.sort(x,axis=0)[::-1]
    nu = (np.cumsum(mu)-k)/range(1,d+1)
    I = [i for i in range(0,d) if mu[i]>nu[i]]
    theta = nu[I[-1]]
    w = np.maximum(x-theta,0)
    return w

''' Define parameter flags '''
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('seed', 1, """Seed. """)
tf.app.flags.DEFINE_float('p_alpha', 1.0, """Imbalance regularization param. """)
tf.app.flags.DEFINE_float('p_lambda', 1e-4, """Weight decay regularization parameter. """)
tf.app.flags.DEFINE_integer('n_in', 3, """Number of representation layers. """)
tf.app.flags.DEFINE_integer('n_out', 3, """Number of regression layers. """)
tf.app.flags.DEFINE_string('activation', 'elu', """Kind of non-linearity. Default relu. """)
tf.app.flags.DEFINE_float('lrate', 1e-3, """Learning rate. """)
tf.app.flags.DEFINE_integer('batch_size', 100, """Batch size. """)
tf.app.flags.DEFINE_integer('dim_in', 200, """Pre-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('dim_out', 100, """Post-representation layer dimensions. """)
tf.app.flags.DEFINE_integer('batch_norm', 0, """Whether to use batch normalization. """)
tf.app.flags.DEFINE_string('normalization', 'divide', """How to normalize representation (after batch norm). none/bn_fixed/divide/project """)
tf.app.flags.DEFINE_string('imb_fun', 'wass', """Which imbalance penalty to use (mmd_lin/mmd_rbf/mmd2_lin/mmd2_rbf/lindisc/wass). """)
tf.app.flags.DEFINE_integer('experiments', 1, """Number of experiments. """)
tf.app.flags.DEFINE_boolean('reweight_sample', 1, """Whether to reweight sample for prediction loss with average treatment probability. """)
tf.app.flags.DEFINE_boolean('split_output', 0, """Whether to split output layers between treated and control. """)
tf.app.flags.DEFINE_integer('rep_weight_decay', 0, """Whether to penalize representation layers with weight decay""")
tf.app.flags.DEFINE_float('dropout_in', 1.0, """Input layers dropout keep rate. """)
tf.app.flags.DEFINE_float('dropout_out', 1.0, """Output layers dropout keep rate. """)
tf.app.flags.DEFINE_float('rbf_sigma', 0.1, """RBF MMD sigma """)
tf.app.flags.DEFINE_float('lrate_decay', 0.97, """Decay of learning rate every 100 iterations """)
tf.app.flags.DEFINE_float('decay', 0.3, """RMSProp decay. """)
tf.app.flags.DEFINE_string('optimizer', 'Adam', """Which optimizer to use. (RMSProp/Adagrad/GradientDescent/Adam)""")
tf.app.flags.DEFINE_float('wass_lambda', 10.0, """Wasserstein lambda. """)
tf.app.flags.DEFINE_integer('wass_iterations', 10, """Number of iterations in Wasserstein computation. """)
tf.app.flags.DEFINE_integer('wass_bpt', 1, """Backprop through T matrix? """)
tf.app.flags.DEFINE_integer('use_p_correction', 0, """Whether to use population size p(t) in mmd/disc/wass.""")
tf.app.flags.DEFINE_integer('iterations', 3000, """Number of iterations. """)
tf.app.flags.DEFINE_float('weight_init', 0.1, """Weight initialization scale. """)
tf.app.flags.DEFINE_string('outdir', 'results/example_ihdp', """Output directory. """)
tf.app.flags.DEFINE_string('datadir', 'data/', """Data directory. """)
tf.app.flags.DEFINE_string('dataform', 'ihdp_npci_1-100.train.npz', """Training data filename form. """)
tf.app.flags.DEFINE_string('data_test', 'ihdp_npci_1-100.test.npz', """Test data filename form. """)
tf.app.flags.DEFINE_integer('pred_output_delay', 200, """Number of iterations between prediction outputs. (-1 gives no intermediate output). """)
tf.app.flags.DEFINE_string('loss', 'l2', """Which loss function to use (l1/l2/log)""")
tf.app.flags.DEFINE_integer('sparse', 0, """Whether data is stored in sparse format (.x, .y). """)
tf.app.flags.DEFINE_integer('varsel', 0, """Whether the first layer performs variable selection. """)
tf.app.flags.DEFINE_integer('repetitions', 1, """Repetitions with different seed.""")
tf.app.flags.DEFINE_float('val_part', 0.3, """Validation part. """)
tf.app.flags.DEFINE_integer('output_delay', 100, """Number of iterations between log/loss outputs. """)

def main():

    ''' Set random seeds '''
    random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    ''' Load dataset '''
    exp = 0
    trainDB, validDB, testDB, _ = get_DataFrame(exp=exp, path='data/IHDP_df.npz')
    n, x_dim = trainDB.x.shape

    ''' Start Session '''
    sess = tf.Session()

    ''' Define model graph '''
    CFR = Net(n, x_dim, FLAGS)

    ''' Set up optimizer '''
    global_step = tf.Variable(0, trainable=False)
    NUM_ITERATIONS_PER_DECAY = 100
    lr = tf.train.exponential_decay(FLAGS.lrate, global_step, NUM_ITERATIONS_PER_DECAY, FLAGS.lrate_decay, staircase=True)
    opt = tf.train.AdamOptimizer(lr)
    train_step = opt.minimize(CFR.tot_loss,global_step=global_step)

    ''' Compute treatment probability'''
    p_treated = np.mean(trainDB.t)

    ''' Set up loss feed_dicts'''
    dict_factual = {CFR.x: trainDB.x, CFR.t: trainDB.t, CFR.y_: trainDB.yf, 
                    CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.r_alpha: FLAGS.p_alpha, CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated}
    dict_valid   = {CFR.x: validDB.x, CFR.t: validDB.t, CFR.y_: validDB.yf, 
                    CFR.do_in: 1.0, CFR.do_out: 1.0, CFR.r_alpha: FLAGS.p_alpha, CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated}
    dict_cfactual = {CFR.x: trainDB.x, CFR.t: 1-trainDB.t, CFR.y_: trainDB.yf, 
                     CFR.do_in: 1.0, CFR.do_out: 1.0}

    ''' Initialize TensorFlow variables '''
    sess.run(tf.global_variables_initializer())

    ''' Train for multiple iterations '''
    objnan = False
    for i in range(FLAGS.iterations):

        ''' Fetch sample '''
        I = random.sample(range(0, n), FLAGS.batch_size)
        x_batch = trainDB.x[I,:]
        t_batch = trainDB.t[I]
        y_batch = trainDB.yf[I]

        ''' Do one step of gradient descent '''
        if not objnan:
            sess.run(train_step, feed_dict={CFR.x: x_batch, CFR.t: t_batch, \
                CFR.y_: y_batch, CFR.do_in: FLAGS.dropout_in, CFR.do_out: FLAGS.dropout_out, \
                CFR.r_alpha: FLAGS.p_alpha, CFR.r_lambda: FLAGS.p_lambda, CFR.p_t: p_treated})

        ''' Project variable selection weights '''
        if FLAGS.varsel:
            wip = simplex_project(sess.run(CFR.weights_in[0]), 1)
            sess.run(CFR.projection, feed_dict={CFR.w_proj: wip})

        ''' Compute loss every N iterations '''
        if i % FLAGS.output_delay == 0 or i==FLAGS.iterations-1:
            obj_loss,f_error,imb_err = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist],feed_dict=dict_factual)
            valid_obj, valid_f_error, valid_imb = sess.run([CFR.tot_loss, CFR.pred_loss, CFR.imb_dist], feed_dict=dict_valid)
            cf_error = sess.run(CFR.pred_loss, feed_dict=dict_cfactual)

            if np.isnan(obj_loss):
                print('Experiment {}: Objective is NaN. Skipping.'.format(exp))
                objnan = True

            loss_str = str(i) + '\tObj: %.3f,\tF: %.3f,\tCf: %.3f,\tImb: %.2g,\tVal: %.3f,\tValImb: %.2g,\tValObj: %.2f' \
                        % (obj_loss, f_error, cf_error, imb_err, valid_f_error, valid_imb, valid_obj)
            print(loss_str)

if __name__ == '__main__':
    main()