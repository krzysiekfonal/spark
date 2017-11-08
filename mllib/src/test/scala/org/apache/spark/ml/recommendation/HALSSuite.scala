package org.apache.spark.ml.recommendation

import org.apache.spark.{SparkFunSuite}
import org.apache.spark.internal.Logging
import org.apache.spark.ml.recommendation.HALS.{Rating}
import org.apache.spark.ml.util.DefaultReadWriteTest
import org.apache.spark.mllib.util.MLlibTestSparkContext
import org.apache.spark.rdd.RDD
import scala.collection.Map

class HALSSuite extends SparkFunSuite with MLlibTestSparkContext with DefaultReadWriteTest with Logging {
  /**
    * Train ALS using the given training set and parameters
    * @param training training dataset
    * @param rank rank of the matrix factorization
    * @param maxIter max number of iterations
    * @param maxInternIter max number of internal iterations
    * @param regParam regularization constant
    * @param tol tolerance
    * @return a trained ALSModel
    */
  def trainHALS(
                training: RDD[Rating[Int]],
                rank: Int,
                partitions: Int,
                maxIter: Int,
                maxInternIter: Int,
                regParam: Double,
                tol: Double): (Array[Rating[Int]], Map[Int, Array[Float]], Map[Int, Array[Float]]) = {
    val(userFactors, itemFactors) = HALS.train(training, rank, partitions, maxIter, maxInternIter, regParam, tol)
    val result = Array.ofDim[Rating[Int]](userFactors.size * itemFactors.size)
    var i: Int = 0
    userFactors.foreach(q => {
      val (k1, v1) = q
      itemFactors.foreach(q => {
        val (k2, v2) = q
        result.update(i, new Rating[Int](k1, k2, (v1,v2).zipped.map(_ * _).sum))
        i+=1
      })
    })

    (result, userFactors, itemFactors)
  }

  def calcSIR(ratings: RDD[Rating[Int]],
              userFactors: Map[Int, Array[Float]],
              itemFactors: Map[Int, Array[Float]]) = {
    Math.sqrt(
      ratings.aggregate(0.0)(
        (acc, q)=>{
          acc + Math.pow((q.rating - ((userFactors(q.user),itemFactors(q.item)).zipped.map(_ * _).sum)), 2)
        },
        (acc1, acc2)=>acc1 + acc2)
    )
  }

  test("Initial test") {
    val trainingSet = sc.parallelize(Array(
      Rating(0, 0, 1), Rating(0, 1, 1), Rating(0, 2, 3), Rating(0, 3, 2),
      Rating(1, 1, 1), Rating(1, 2, 1), Rating(1, 3, 2),
      Rating(2, 0, 1), Rating(2, 2, 1),
      Rating(3, 0, 2), Rating(3, 1, 1), Rating(3, 2, 5), Rating(3, 3, 2)
    )) //4x4
    val (result, userFactors, itemFactors) =
      trainHALS(trainingSet, rank = 4, partitions = 2, maxIter = 100, maxInternIter = 10, regParam = 0.01, tol = 1e-12)

    println(calcSIR(trainingSet, userFactors, itemFactors))
  }

  test("HALS.Matrix") {
    val m1 = new HALS.Matrix(3, 3)

    //verify updates in matrix
    m1(1,1) = 4.0f
    m1.update(2, 2, 2.0f)
    assert(m1(1,1)== 4.0f)
    assert(m1(2,2) == 2.0f)

    //verify updateAdd
    m1.updateAdd(2,2, 1.0f)
    assert(m1(2,2) == 3.0f)

    //verify add
    val m2 = new HALS.Matrix(3, 3)
    m2(0,0) = 2.0f
    m2(1,1) = -1.0f
    m2(2,2) = 1.0f
    m1.add(m2)
    assert(m1(0,0) == 2.0f)
    assert(m1(1,1) == 3.0f)
    assert(m1(2,2) == 4.0f)

    //verify scalarRowMultiply
    m1(1,0) = 1.0f
    val expectedVal = 8.0f
    assert(m1.scalarRowMultiply(1, Array(2.0f, 2.0f, 2.0f)) == expectedVal)
  }
}