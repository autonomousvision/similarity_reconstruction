#include "svm_wrapper.h"
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <string.h>
#include <strings.h>
#include <stdexcept>
#include <numeric>
#include "third_party/libsvm/libsvm-3.20/svm.h"
#include <glog/logging.h>

#define CMD_LEN 2048
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
int print_null(const char *s, ...) {return 0;}
static int (*info)(const char *fmt,...) = &printf;

int exit_with_help()
{
    printf(
                "Usage: svm-predict [options] test_file model_file output_file\n"
                "options:\n"
                "-b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); for one-class SVM only 0 is supported\n"
                "-q : quiet mode (no outputs)\n"
                );
    return -1;
}

void SVMWrapper::do_cross_validation()
{
    int i;
    int total_correct = 0;
    double total_error = 0;
    double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
    double *target = Malloc(double,prob.l);

    svm_cross_validation(&prob,&param,nr_fold,target);
    if(param.svm_type == EPSILON_SVR ||
            param.svm_type == NU_SVR)
    {
        for(i=0;i<prob.l;i++)
        {
            double y = prob.y[i];
            double v = target[i];
            total_error += (v-y)*(v-y);
            sumv += v;
            sumy += y;
            sumvv += v*v;
            sumyy += y*y;
            sumvy += v*y;
        }
        printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
        printf("Cross Validation Squared correlation coefficient = %g\n",
               ((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
               ((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
               );
    }
    else
    {
        for(i=0;i<prob.l;i++)
            if(target[i] == prob.y[i])
                ++total_correct;
        printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
    }
    free(target);
}

void SVMWrapper::Clear()
{
    // training
    svm_destroy_param(&param);
    svm_free_and_destroy_model(&model);
    if (prob.x) free(prob.x);
    prob.x = NULL;
    if (prob.y) free(prob.y);
    prob.y = NULL;
    if (x_space) free(x_space);
    x_space = NULL;
    cross_validation = 0;
    nr_fold = 0;

    // testing
    // x.clear();
    // predict_probability=0;
}

bool SVMWrapper::LoadSVMModel(const std::string &model_path)
{
    Clear();
    model = svm_load_model(model_path.c_str());
    if (model == NULL) {
        LOG(WARNING) << "Reading file: " << model_path << "failed.";
        return false;
    }
    model_path_ = model_path;
    return true;
}

bool SVMWrapper::SaveSVMModel(const std::string &save_path) const
{
    return svm_save_model(save_path.c_str(), model) >= 0;
}

int SVMWrapper::SVMTrain(const std::vector<std::vector<float> > &features, const std::vector<int> &labels, const std::string &options, const std::string* save_path)
{
    Clear();
    std::string save_fullfile;
    if (save_path)
    {
         save_fullfile = *save_path + "_svm_model";
    }
    const char *error_msg = NULL;

    OptionStringToSVMParam(options);
    VectorFeaturesToSVMProblem(features, labels);
    error_msg = svm_check_parameter(&prob,&param);

    if(error_msg)
    {
        fprintf(stderr,"ERROR: %s\n",error_msg);
        exit(1);
    }

    if(cross_validation)
    {
        do_cross_validation();
    }
    else
    {
        model = svm_train(&prob,&param);
        if(svm_save_model(save_fullfile.c_str(),model))
        {
            fprintf(stderr, "can't save model to file %s\n", save_fullfile.c_str());
            return -1;
        }
        //svm_free_and_destroy_model(&model);
    }
    ConvertToPrimalForm(features[0].size());
    return 0;
}

int SVMWrapper::SVMTrain(const Eigen::SparseMatrix<float> &features, const std::vector<char> &labels, const std::string &options, const std::string &save_path)
{
    fprintf(stderr, "begin training\n");
    Clear();
    std::string save_fullfile = save_path + "_svm_model";
    const char *error_msg = NULL;

    OptionStringToSVMParam(options);
    SparseFeaturesToSVMProblem(features, labels);
    error_msg = svm_check_parameter(&prob,&param);

    if(error_msg)
    {
        fprintf(stderr,"ERROR: %s\n",error_msg);
        exit(1);
    }

    if(cross_validation)
    {
        do_cross_validation();
    }
    else
    {
        model = svm_train(&prob,&param);
        //if(svm_save_model(save_fullfile.c_str(),model))
        //{
        //    fprintf(stderr, "can't save model to file %s\n", save_fullfile.c_str());
        //    return -1;
        //}
        //svm_free_and_destroy_model(&model);
    }
    ConvertToPrimalForm(features.rows());
    return 0;
}

int SVMWrapper::SVMPredict(const std::vector<std::vector<float>> &features, const std::vector<int>& input_labels, const std::string & options,
                           std::vector<int>* labels, std::vector<std::vector<float>> * scores, float* accuracy, const std::string* save_path) const
{
    int argc = 1;
    char cmd[CMD_LEN];
    char *argv[CMD_LEN/2];
    strcpy(cmd, options.c_str());
    if((argv[argc] = strtok(cmd, " ")) != NULL)
        while((argv[++argc] = strtok(NULL, " ")) != NULL)
            ;

    int i;
    int predict_probability = 0;
    // parse options
    for(i=1;i<argc;i++)
    {
        if(argv[i][0] != '-') break;
        ++i;
        switch(argv[i-1][1])
        {
        case 'b':
            predict_probability = atoi(argv[i]);
            break;
        case 'q':
            info = &print_null;
            i--;
            break;
        default:
            fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
            return exit_with_help();
        }
    }

    if(predict_probability)
    {
        if(svm_check_probability_model(model)==0)
        {
            fprintf(stderr,"Model does not support probabiliy estimates\n");
            return -1;
        }
    }
    else
    {
        if(svm_check_probability_model(model)!=0)
            info("Model supports probability estimates, but disabled in prediction.\n");
    }

    SVMPredictImpl(features, input_labels, predict_probability, labels, scores, accuracy, save_path);
    return 0;

}

int SVMWrapper::SVMPredict_Primal(const std::vector<std::vector<float> > &features, const std::vector<int> &input_labels, const std::string &options, std::vector<int> *labels, std::vector<std::vector<float> > *scores, float *accuracy, const std::string *save_path) const
{

    assert(model->nr_class == 2);
    labels->resize(features.size());
    scores->resize(features.size());
    int correct_sample = 0;
    for (int i = 0; i < features.size(); ++i)
    {
        float score = std::inner_product(features[i].begin(), features[i].end(), weights.begin(), 0.f) + bias;
        (*labels)[i] = score > 0 ? 1 : -1;
        (*scores)[i].resize(1);
        (*scores)[i][0] = score;
        if ((*labels)[i] == input_labels[i])
            correct_sample++;
    }
    *accuracy = (float)correct_sample/features.size();
    return 0;
}

int SVMWrapper::SVMPredict_Primal(const std::vector<Eigen::SparseVector<float> > &features, const std::vector<int> &input_labels, const std::string &options, std::vector<int> *labels, std::vector<std::vector<float> > *scores, float *accuracy, const std::string *save_path) const
{
    assert(model->nr_class == 2);
    labels->resize(features.size());
    scores->resize(features.size());
    int correct_sample = 0;
    for (int i = 0; i < features.size(); ++i)
    {
        float score = 0;
        for (Eigen::SparseVector<double>::InnerIterator it(features[i]); it; ++it)
        {
          score += (it.value() * weights[it.index()]);
        }
        score += bias;
        (*labels)[i] = score > 0 ? 1 : -1;
        (*scores)[i].resize(1);
        (*scores)[i][0] = score;
        if ((*labels)[i] == input_labels[i])
            correct_sample++;
    }
    *accuracy = (float)correct_sample/features.size();
    return 0;
}

int SVMWrapper::SVMPredict_Primal(const Eigen::SparseVector<float> &feature, float* score, char* label) const
{
    assert(model->nr_class == 2);
    float cur_score = 0;
    for (Eigen::SparseVector<float>::InnerIterator it(feature); it; ++it)
    {
        cur_score += (it.value() * weights[it.index()]);
    }
    cur_score += bias;
    *label = cur_score > 0 ? 1 : -1;
    *score = cur_score;
    return 0;
}

void SVMWrapper::ConvertToPrimalForm(int feat_dim)
{
    const int number_SVs = model->l;
    weights.assign(feat_dim, 0.f);
    bias = -(model->rho[0]);
    // compute weights
    // get SVs
    std::vector<std::vector<double>> SVs(number_SVs);
    for (int i = 0; i < SVs.size(); ++i)
    {
        SVs[i].assign(feat_dim, 0.f);
    }
    if (model->param.kernel_type == PRECOMPUTED || model->nr_class != 2)
    {
        throw std::runtime_error("unhandled kernel type or other svm models");
    }
    for(int i = 0; i < number_SVs; i++)
    {
        int x_index = 0;
        while (model->SV[i][x_index].index != -1)
        {
            int cur_index = model->SV[i][x_index].index - 1;
            SVs[i][cur_index] = model->SV[i][x_index].value;
            assert(cur_index < feat_dim);
            //CHECK_LT(cur_index, feat_dim);
            x_index++;
        }
    }
    for (int i = 0; i < number_SVs; ++i)
    {
        for (int j = 0; j < feat_dim; ++j)
        {
            double cur_coeff = (model->sv_coef[0][i]);
            weights[j] += (SVs[i][j] * cur_coeff);
        }
    }
    return;
}

int SVMWrapper::SVMPredictImpl(const std::vector<std::vector<float> > &input_features,  const std::vector<int>& input_labels, const int predict_probability,
                               std::vector<int> *output_labels, std::vector<std::vector<float>> * scores, float *output_accuracy, const std::string* save_path) const
{
    FILE* output = NULL;
    if (save_path)
    {
        output = fopen(save_path->c_str(), "w");
    }
    else
    {
        output = fopen("tmp", "w");
    }

    int correct = 0;
    int total = 0;
    double error = 0;
    double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;   

    int svm_type=svm_get_svm_type(model);
    int nr_class=svm_get_nr_class(model);
    double *prob_estimates = NULL;

    double *dec_values = NULL;
    if(model->param.svm_type == ONE_CLASS ||
       model->param.svm_type == EPSILON_SVR ||
       model->param.svm_type == NU_SVR)
        dec_values = Malloc(double, 1);
    else
        dec_values = Malloc(double, nr_class*(nr_class-1)/2);

    output_labels->resize(input_features.size());
    scores->resize(input_features.size());
    if(svm_type == ONE_CLASS ||
          svm_type == EPSILON_SVR ||
          svm_type == NU_SVR ||
          nr_class == 1)
    {
        for (int i = 0; i < scores->size(); ++i)
        {
            (*scores)[i].resize(1);
        }
    }
    else
    {
        for (int i = 0; i < scores->size(); ++i)
        {
            (*scores)[i].resize(nr_class*(nr_class-1)/2);
        }
    }

    if(predict_probability)
    {
        if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
            info("Prob. model for test data: target value = predicted value + z,\nz: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma=%g\n",svm_get_svr_probability(model));
        else
        {
            int *labels=(int *) malloc(nr_class*sizeof(int));
            svm_get_labels(model,labels);
            prob_estimates = (double *) malloc(nr_class*sizeof(double));
            fprintf(output,"labels");
            for(int j=0;j<nr_class;j++)
                fprintf(output," %d",labels[j]);
            fprintf(output,"\n");
            free(labels);
        }
    }

    output_labels->resize(input_features.size());
    for (int samplei = 0; samplei < input_features.size(); ++samplei)
    {
        double target_label, predict_label;
        target_label = input_labels[samplei];
        std::vector<struct svm_node> x;
        //x.clear();
        for (int idx = 0; idx < input_features[samplei].size(); ++idx)
        {
            if (input_features[samplei][idx] != 0)
            {
                struct svm_node curnode;
                curnode.index = idx + 1;
                curnode.value = input_features[samplei][idx];
                x.push_back(curnode);
            }
        }
        struct svm_node curnode;
        x.push_back(curnode);
        x.back().index = -1;

        if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))
        {
            predict_label = svm_predict_probability(model, &(x[0]),prob_estimates);
            fprintf(output,"%g",predict_label);
            for(int j=0;j<nr_class;j++)
                fprintf(output," %g",prob_estimates[j]);
            fprintf(output,"\n");
            for(int i=0;i<nr_class;i++)  // may have problem?
                (*scores)[samplei][i] = prob_estimates[i];
        }
        else
        {
            predict_label = svm_predict_values(model,&(x[0]), dec_values);
            fprintf(output,"%g\n",predict_label);
            for(int i=0;i<(nr_class*(nr_class-1))/2;i++)
                (*scores)[samplei][i] = dec_values[i];
        }

        (*output_labels)[samplei] = predict_label;
        if(predict_label == target_label)
            ++correct;
        error += (predict_label-target_label)*(predict_label-target_label);
        sump += predict_label;
        sumt += target_label;
        sumpp += predict_label*predict_label;
        sumtt += target_label*target_label;
        sumpt += predict_label*target_label;
        ++total;
    }

    if (svm_type==NU_SVR || svm_type==EPSILON_SVR)
    {
        info("Mean squared error = %g (regression)\n",error/total);
        info("Squared correlation coefficient = %g (regression)\n",
             ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
             ((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))
             );
    }
    else
        info("Accuracy = %g%% (%d/%d) (classification)\n",
             (double)correct/total*100,correct,total);
    if(predict_probability)
        free(prob_estimates);
    *output_accuracy = (double)correct/total*100,correct,total;
    fclose(output);
    free(dec_values);
    return 0;
}

int SVMWrapper::VectorFeaturesToSVMProblem(const std::vector<std::vector<float> > &samples, const std::vector<int>& labels)
{
    // using size_t due to the output type of matlab functions
    size_t i, j, k, l;
    size_t elements, max_index, sc, label_vector_row_num;
    assert(!samples.empty());

    if (prob.x) free(prob.x);
    prob.x = NULL;
    if (prob.y) free(prob.y);
    prob.y = NULL;
    if (x_space) free(x_space);
    x_space = NULL;

    sc = samples[0].size();

    elements = 0;
    l = samples.size();
    label_vector_row_num = labels.size();
    prob.l = (int)l;

    if(label_vector_row_num!=l)
    {
        printf("Length of label vector does not match # of instances.\n");
        return -1;
    }

    if(param.kernel_type == PRECOMPUTED)
        elements = l * (sc + 1);
    else
    {
        for(i = 0; i < l; i++)
        {
            for(k = 0; k < sc; k++)
                if(samples[i][k] != 0)
                    elements++;
            // count the '-1' element
            elements++;
        }
    }

    prob.y = Malloc(double,l);
    prob.x = Malloc(struct svm_node *,l);
    x_space = Malloc(struct svm_node, elements);

    max_index = sc;
    j = 0;
    for(i = 0; i < l; i++)
    {
        prob.x[i] = &x_space[j];
        prob.y[i] = labels[i];

        for(k = 0; k < sc; k++)
        {
            if(param.kernel_type == PRECOMPUTED || samples[i][k] != 0)
            {
                x_space[j].index = (int)k + 1;
                x_space[j].value = samples[i][k];
                j++;
            }
        }
        x_space[j++].index = -1;
    }

    if(param.gamma == 0 && max_index > 0)
        param.gamma = (double)(1.0/max_index);

    if(param.kernel_type == PRECOMPUTED)
        for(i=0;i<l;i++)
        {
            if((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > (int)max_index)
            {
                printf("Wrong input format: sample_serial_number out of range\n");
                return -1;
            }
        }
    return 0;
}

int SVMWrapper::SparseFeaturesToSVMProblem(const Eigen::SparseMatrix<float> &samples, const std::vector<char> &labels)
{
    CHECK_NE(param.kernel_type, PRECOMPUTED);  // not supported in this wrapper
    // samples size: feat_dim * sample_num
    // using size_t due to the output type of matlab functions
    size_t i, j, k, l;
    size_t elements, max_index, sc, label_vector_row_num;

    if (prob.x) free(prob.x);
    prob.x = NULL;
    if (prob.y) free(prob.y);
    prob.y = NULL;
    if (x_space) free(x_space);
    x_space = NULL;

    sc = samples.rows();

    elements = 0;
    l = samples.cols();  // l: sample number
    label_vector_row_num = labels.size();
    prob.l = (int)l;

    if(label_vector_row_num!=l)
    {
        printf("Length of label vector does not match # of instances.\n");
        return -1;
    }

    elements = samples.nonZeros();
    prob.y = Malloc(double,l);
    prob.x = Malloc(struct svm_node *,l);
    x_space = Malloc(struct svm_node, elements + l);

    max_index = sc;
    j = 0;
    for (int i = 0; i < samples.outerSize(); ++i)
    {
        prob.x[i] = &x_space[j];
        prob.y[i] = labels[i];
        for (Eigen::SparseMatrix<float>::InnerIterator it(samples, i); it; ++it)
        {
            x_space[j].index = it.row() + 1;
            x_space[j].value = it.value();
            j++;
        }
        x_space[j++].index = -1;
    }

    if(param.gamma == 0 && max_index > 0)
        param.gamma = (double)(1.0/max_index);

    return 0;
}

int SVMWrapper::OptionStringToSVMParam(const std::string &options)
{
    int i, argc = 1;

    char cmd[CMD_LEN];
    char *argv[CMD_LEN/2];
    int (*print_func)(const char *, ...) = NULL;

    // default values
    param.svm_type = C_SVC;
    param.kernel_type = RBF;
    param.degree = 3;
    param.gamma = 0;	// 1/num_features
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 1;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;
    cross_validation = 0;


    // put options in argv[]
    strcpy(cmd, options.c_str());
    if((argv[argc] = strtok(cmd, " ")) != NULL)
        while((argv[++argc] = strtok(NULL, " ")) != NULL)
            ;

    // parse options
    for(i=1;i<argc;i++)
    {
        if(argv[i][0] != '-') break;
        ++i;
        if(i>=argc && argv[i-1][1] != 'q')	// since option -q has no parameter
            return 1;
        switch(argv[i-1][1])
        {
        case 's':
            param.svm_type = atoi(argv[i]);
            break;
        case 't':
            param.kernel_type = atoi(argv[i]);
            break;
        case 'd':
            param.degree = atoi(argv[i]);
            break;
        case 'g':
            param.gamma = atof(argv[i]);
            break;
        case 'r':
            param.coef0 = atof(argv[i]);
            break;
        case 'n':
            param.nu = atof(argv[i]);
            break;
        case 'm':
            param.cache_size = atof(argv[i]);
            break;
        case 'c':
            param.C = atof(argv[i]);
            break;
        case 'e':
            param.eps = atof(argv[i]);
            break;
        case 'p':
            param.p = atof(argv[i]);
            break;
        case 'h':
            param.shrinking = atoi(argv[i]);
            break;
        case 'b':
            param.probability = atoi(argv[i]);
            break;
        case 'q':
            print_func = &print_null;
            i--;
            break;
        case 'v':
            cross_validation = 1;
            nr_fold = atoi(argv[i]);
            if(nr_fold < 2)
            {
                printf("n-fold cross validation: n must >= 2\n");
                return 1;
            }
            break;
        case 'w':
            ++param.nr_weight;
            param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
            param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
            param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
            param.weight[param.nr_weight-1] = atof(argv[i]);
            break;
        default:
            printf("Unknown option -%c\n", argv[i-1][1]);
            return 1;
        }
    }

    printf("svmwrapper: svm_c: %f\n", param.C);
    printf("svmwrapper: param weight: %f %f\n", param.weight[0], param.weight[1]);
    svm_set_print_string_function((void (*)(const char*))print_func);

    return 0;
}
