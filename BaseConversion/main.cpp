#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <bitset>
#include <string>
#include <random>
#include <Eigen/Dense>

//this values changes based on the dimension of the decoded output, higher values=lower loss, the relationship between them is explained in the pdf
const int MAX_VALUE = 32;
const int DIM = 16;
//this values are empirically computed to maximize the precision of the encoding, we'll talk mor4 about this in the pdf
const int MAX_VALUE_x = 1540;
const int MAX_VALUE_y = 1420;
const int MAX_VALUE_cov = 70;

const int VECTOR_SIZE = 3;

using Matrix2f = Eigen::Matrix<float, 2, 2>;
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<float> dis1(0, MAX_VALUE_x);
std::uniform_real_distribution<float> dis3(0, MAX_VALUE_y);
std::uniform_real_distribution<float> dis2(-MAX_VALUE_cov, MAX_VALUE_cov);

//encode a given matrix into a unique string
std::string encode(Matrix2f mat)
{
  //keeps track of the sign of the covXY
  char segno;
  if (mat(0, 1) < 0)
  {
    mat(0, 1) = -mat(0, 1);
    segno = '1';
  }
  else
  {
    segno = '0';
  }

  //convert the matrix into a vector of 3 values
  Eigen::Vector3i vec;
  vec << std::min(static_cast<int>(mat(0, 0)), MAX_VALUE),
         std::min(static_cast<int>(mat(0, 1)), MAX_VALUE),
         std::min(static_cast<int>(mat(1, 1)), MAX_VALUE);

  int x = 0;
  for (int i = 0; i < VECTOR_SIZE; ++i)
  {
    x += vec(i) * pow(MAX_VALUE + 1, VECTOR_SIZE - 1 - i);
  }

  std::bitset<DIM> binary(x);
  return segno + binary.to_string();
}

Matrix2f decode(std::string x)
{
  char segno = x[0];//extracting the sign of the covXY
  int value = std::stoi(x.substr(1), nullptr, 2);

  //converting the value into a vector of 3 values
  Eigen::Vector3i vec;
  for (int i = 0; i < VECTOR_SIZE; ++i)
  {
    vec(i) = value % (MAX_VALUE + 1);
    value /= (MAX_VALUE + 1);
  }

  if (segno == '1')
    vec(1) = -vec(1);

  Matrix2f mat;
  mat << vec(2), vec(1), vec(1), vec(0);
  return mat;
}

Matrix2f normalize(Matrix2f mat, int maxX, int maxY, int maxCov)
{
  mat(0, 0) = std::round((mat(0, 0) / maxX)*MAX_VALUE);
  mat(1, 0) = std::round((mat(1, 0) / maxCov)*MAX_VALUE);
  mat(0, 1) = std::round((mat(0, 1) / maxCov)*MAX_VALUE);
  mat(1, 1) = std::round((mat(1, 1) / maxY)*MAX_VALUE);

  return mat;
}

Matrix2f denormalize(Matrix2f mat, int maxX, int maxY, int maxCov)
{ 
  mat(0, 0) = std::round((mat(0, 0) * maxX) / MAX_VALUE);
  mat(1, 0) = std::round((mat(1, 0) * maxCov) / MAX_VALUE);
  mat(0, 1) = std::round((mat(0, 1) * maxCov) / MAX_VALUE);
  mat(1, 1) = std::round((mat(1, 1) * maxY) / MAX_VALUE);
  return mat;
}

int main()
{
  //generate a random matrix2f
  Matrix2f mat_o;
  mat_o << std::round(dis1(gen)), std::round(dis2(gen)), std::round(dis2(gen)), std::round(dis3(gen));
  mat_o(0, 1) = mat_o(1, 0);
  std::cout << "Original matrix: " << std::endl
            << mat_o << std::endl << std::endl;

  //normalize the matrix
  Matrix2f mat = normalize(mat_o, MAX_VALUE_x, MAX_VALUE_y, MAX_VALUE_cov);
  std::cout << "Normalized matrix: " << std::endl
            << mat << std::endl << std::endl;

  //encode the matrix into a string
  std::string encoded = encode(mat);
  std::cout << "Encoded: " << encoded << std::endl << std::endl;

  //decode the string into a matrix
  mat = decode(encoded);
  std::cout << "Decoded matrix: " << std::endl
            << mat << std::endl << std::endl;

  //denormalize the matrix
  mat = denormalize(mat, MAX_VALUE_x, MAX_VALUE_y, MAX_VALUE_cov);
  std::cout << "denormalized matrix: " << std::endl
            << mat << std::endl << std::endl;

  //compute mean squared error
  double loss = (mat_o - mat).array().square().sum() / (mat_o.rows() * mat.cols());
  std::cout << "MSE: " << loss << std::endl;
  return 0;
}