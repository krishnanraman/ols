package ols
import org.apache.commons.math3.fitting.{PolynomialCurveFitter=>PCF}
import org.apache.commons.math3.fitting.{WeightedObservedPoint => Point}
import org.apache.commons.math3.distribution.TDistribution
import scala.collection.JavaConverters._

object LinearRegression extends App {
  type D = Double

  println("Random Dataset around 2x + 3")
  val xy = (0.0 to 100.0 by 0.2).map{ x=> (x,2 * x + 3 + math.random) }
  val ols = LinearRegression(xy)
  println("Model: " + ols.fit.format(3).mkStr)
  println( ols.predict(4.0))
}

import LinearRegression.D
case class ConfidenceInterval(lowerBound:D, upperBound:D)
case class Prediction(value:D, ci:ConfidenceInterval)

trait regress {
  def fit: Line
  def predict(x:D): Prediction
}

case class LinearRegression(xy:Seq[(D,D)]) extends regress {
  assert(xy.size > 2)

  private[this] var line:Option[Line] = None
  private val n = xy.size

  def fit:Line = {
      val poly = PCF.create(1)
      val res = poly.fit( xy.map { pt =>
        val (x,y) = pt
        new Point(1.0, x, y)
      }.asJava )
      line = Some(Line(res(1), res(0)))
      line.get
  }

  def yMean = xy.map { case (x,y) => y}.sum/(n+0.0)

  def xMean = xy.map { case (x,y) => x}.sum/(n+0.0)

  // aka standard error, root mean square error, rmse, ...
  def se = standardErrorResiduals
  def rmse = standardErrorResiduals
  def rootMeanSquareError = standardErrorResiduals

  def standardErrorResiduals = {
    val mse = xy.map { case (x,y) => (y, line.get.apply(x))}
      .map{ case(yi,yhat) => (yi - yhat)*(yi-yhat) }
      .sum / (n - 2)
    math.sqrt(mse)
  }

  def sey = standardErrorPrediction _
  def standardErrorPrediction(xk:D) = {
      se * math.sqrt((1+1.0/n + ((xk - xMean)*(xk-xMean)/xerror)))
  }

  def xerror = {
    xy.map { case (x,y) => (x, xMean)}
      .map{ case(xi,xbar) => (xi - xbar)*(xi-xbar) }
      .sum
  }

  def standardErrorSlope = {
    se/math.sqrt(xerror)
  }

  def degreesOfFreedom = n-2

  def tDist = new TDistribution(degreesOfFreedom)

  def tstat = tDist.inverseCumulativeProbability(0.975) // 95% Confidence Interval, 2-sided test

  def predict(x:D): Prediction = {
    val yvalue = line.get.apply(x)
    val err = tstat * sey(x)
    val lower = yvalue - err
    val upper = yvalue + err
    Prediction(yvalue, ConfidenceInterval(lower, upper))
  }
}

case class Line(slope:D, intercept:D) {

  def format(numberOfDecimals:Int):Line = {
    val formatter = "%."+numberOfDecimals+"f"
    val res = Seq(slope,intercept).map{ x=> formatter.format(x) }.map{ x=> x.toDouble }
    Line(res(0), res(1))
  }

  def mkStr = List(slope, intercept)
  .zip(List("x",""))
  .map { i=> i._1 + i._2}
  .mkString(" + ")
  .replace("+ -", "-")

  def apply = {x:D => slope*x + intercept}
}

