#include "matlab_utility.h"
#include <iostream>
using namespace std;
namespace matlabutility {

void EigenSparseMatrixs2MatlabArray(const Eigen::SparseMatrix<float, Eigen::ColMajor>& mat, mwArray* mwarray) {
    assert((*mwarray).NumberOfElements() == mat.size());
    for (int k=0; k<mat.outerSize(); ++k)
      for (Eigen::SparseMatrix<float>::InnerIterator it(mat,k); it; ++it)
      {
          (*mwarray)(it.row() + 1, it.col() + 1) = it.value();
      }
}

void MatlabArray2EigenMatrix(const mwArray& mwarray, Eigen::MatrixXf* mat) {
    mwArray size = mwarray.GetDimensions();
    mat->resize(size(1), size(2));
    if (!(int(size(1)) > 0 && int(size(2)) > 0)) return;
    mwarray.GetData((mxSingle*)(mat->data()),mwarray.NumberOfElements());
}

void MatlabArray2EigenSparseMatrix(const mwArray& mwarray, Eigen::SparseMatrix<float, Eigen::ColMajor>* mat) {
    Eigen::MatrixXf dmat;
    MatlabArray2EigenMatrix(mwarray, &dmat);
    (*mat) = dmat.sparseView();
}

void MatlabArray2EigenVector(const mwArray& mwarray, Eigen::VectorXf* vec) {
    vec->resize(mwarray.NumberOfElements());
    mwarray.GetData((mxSingle*)vec->data(), mwarray.NumberOfElements());
}

void MatlabArray2EigenSparseVector(const mwArray& mwarray, Eigen::SparseVector<float>* vec) {
    vec->resize(mwarray.NumberOfElements());
    for (int i = 1; i <= mwarray.NumberOfElements(); ++i) {
        if ((double)mwarray(i) != 0) {
        (*vec).coeffRef(i - 1) = (float)mwarray(i);
        }
    }
}

void ConvertCopyMatlabMatrixToEigenVector(Eigen::MatrixXf* mat, const double* matlab_array, const Eigen::Vector2i& bbsize)
{
    const int ty = bbsize[0];
    const int tx = bbsize[1];
    mat->resize(ty,  tx);
    int cnt = 0;
    if (mat->IsRowMajor)
    {
        for(int i=0;i<tx;i++)
            for(int j=0;j<ty;j++)
            {
                (*mat)( j * tx + i ) = matlab_array[cnt];
                cnt++;
            }
    }
    else
    {
        for(int i=0;i<tx;i++)
            for(int j=0;j<ty;j++)
            {
                (*mat)( i * ty + j ) = matlab_array[cnt];
                cnt++;
            }
    }
}

bool ReadMatrixMatlab(const string &filepath, const string &varname, Eigen::MatrixXf *matrix)
{
    Eigen::Vector2i bb_size;
    mat_t *matfp;
    matvar_t *matvar;
    matfp = Mat_Open(filepath.c_str(),MAT_ACC_RDONLY);
    if ( NULL == matfp )
    {
        fprintf(stderr,"Error opening MAT file \"%s\"!\n",filepath.c_str());
        return false;
    }
    matvar = Mat_VarRead(matfp, varname.c_str());
    if ( NULL == matvar )
    {
        fprintf(stderr,"Variable ’%s’ not found, or error "
                       "reading MAT file\n", varname.c_str());
        return false;
    }
    else
    {
        if ( matvar->rank != 2 || matvar->isComplex || matvar->data_type != MAT_T_DOUBLE )
        {
            fprintf(stderr,"Variable ’%s’ is not a valid double matrix. Rank: %d, iscomplex: %d, type: %d\n", varname.c_str(), matvar->rank, int(matvar->isComplex), matvar->data_type );
            fprintf(stderr, "dim y, x: %d %d\n", matvar->dims[0], matvar->dims[1]);
            return false;
        }
        bb_size[0] = matvar->dims[0];
        bb_size[1] = matvar->dims[1];
        ConvertCopyMatlabMatrixToEigenVector(matrix, (double*)matvar->data, bb_size);
        Mat_VarFree(matvar);
    }
    Mat_Close(matfp);
    cout << "matlab mat '" << varname << "' read in finished. " << endl;
    return true;
}

bool WriteMatrixMatlab(const string &save_filepath, const string &varname, const Eigen::MatrixXf &matrix)
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> col_mat = matrix.cast<double>();
    /* setup the output */
    mat_t *mat;
    matvar_t *matvar;
    size_t dims[2] = {col_mat.rows(), col_mat.cols()};
    cout << "saving matlab mat" << endl;
    cout << (save_filepath) << endl;
    mat = Mat_Open((save_filepath).c_str(),MAT_ACC_RDWR);
    if (!mat)
    {
        mat = Mat_Create((save_filepath).c_str(), NULL);
    }
    {
        cout << "opened mat" << endl;
        /* first matrix */
        matvar = Mat_VarCreate(varname.c_str(),MAT_C_DOUBLE,MAT_T_DOUBLE,2,
                               dims, (double*)col_mat.data(),0);
        Mat_VarWrite( mat, matvar, MAT_COMPRESSION_NONE);
        Mat_VarFree(matvar);
        Mat_Close(mat);
        return true;
    }
}

bool WritePCAOutputResult(const string &filepath, Eigen::SparseVector<float> mean_mat, Eigen::SparseVector<float> mean_weight, Eigen::SparseMatrix<float, Eigen::ColMajor> base_mat, Eigen::MatrixXf coeff_mat) {
    using namespace std;
    Eigen::MatrixXf mean_mat_save = mean_mat;
    Eigen::MatrixXf mean_weight_save = mean_weight;
    Eigen::MatrixXf base_mat_save = base_mat;
    cerr << "begin write" << endl;
    WriteMatrixMatlab(filepath, "mean_mat", mean_mat_save);
    WriteMatrixMatlab(filepath, "mean_weight", mean_weight_save);
    WriteMatrixMatlab(filepath, "base_mat", base_mat_save);
    WriteMatrixMatlab(filepath, "coeff_mat", coeff_mat);
    cerr << "finish write" << endl;
    return true;
}

bool ReadPCAOutputResult(const string &filepath, Eigen::SparseVector<float> *mean_mat, Eigen::SparseVector<float> *mean_weight, Eigen::SparseMatrix<float, Eigen::ColMajor> *base_mat, Eigen::MatrixXf *coeff_mat) {
    Eigen::MatrixXf mean_mat_save = *mean_mat;
    Eigen::MatrixXf mean_weight_save = *mean_weight;
    Eigen::MatrixXf base_mat_save = *base_mat;
    ReadMatrixMatlab(filepath, "mean_mat", &mean_mat_save);
    ReadMatrixMatlab(filepath, "mean_weight", &mean_weight_save);
    ReadMatrixMatlab(filepath, "base_mat", &base_mat_save);
    ReadMatrixMatlab(filepath, "coeff_mat", coeff_mat);
    CHECK_EQ(mean_mat_save.cols(), 1);
    CHECK_EQ(mean_weight_save.cols(),1);
    (*mean_mat) = mean_mat_save.col(0).sparseView();
    (*mean_weight) = mean_weight_save.col(0).sparseView();
    (*base_mat) = base_mat_save.sparseView();
}

}
