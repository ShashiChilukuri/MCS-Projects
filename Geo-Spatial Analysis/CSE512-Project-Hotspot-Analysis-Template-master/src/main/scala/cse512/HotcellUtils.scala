package cse512

import java.sql.Timestamp
import java.text.SimpleDateFormat
import java.util.Calendar

object HotcellUtils {
  val coordinateStep = 0.01

  def CalculateCoordinate(inputString: String, coordinateOffset: Int): Int =
  {
    // Configuration variable:
    // Coordinate step is the size of each cell on x and y
    var result = 0
    coordinateOffset match
    {
      case 0 => result = Math.floor((inputString.split(",")(0).replace("(","").toDouble/coordinateStep)).toInt
      case 1 => result = Math.floor(inputString.split(",")(1).replace(")","").toDouble/coordinateStep).toInt
      // We only consider the data from 2009 to 2012 inclusively, 4 years in total. Week 0 Day 0 is 2009-01-01
      case 2 => {
        val timestamp = HotcellUtils.timestampParser(inputString)
        result = HotcellUtils.dayOfMonth(timestamp) // Assume every month has 31 days
      }
    }
    return result
  }

  def timestampParser (timestampString: String): Timestamp =
  {
    val dateFormat = new SimpleDateFormat("yyyy-MM-dd hh:mm:ss")
    val parsedDate = dateFormat.parse(timestampString)
    val timeStamp = new Timestamp(parsedDate.getTime)
    return timeStamp
  }

  def dayOfYear (timestamp: Timestamp): Int =
  {
    val calendar = Calendar.getInstance
    calendar.setTimeInMillis(timestamp.getTime)
    return calendar.get(Calendar.DAY_OF_YEAR)
  }

  def dayOfMonth (timestamp: Timestamp): Int =
  {
    val calendar = Calendar.getInstance
    calendar.setTimeInMillis(timestamp.getTime)
    return calendar.get(Calendar.DAY_OF_MONTH)
  }

  def squared(a:Int):Double=
  {
    return (a*a).toDouble
  }

  def CellInBounds(x:Double, y:Double, z:Int, minX:Double, maxX:Double, minY:Double, maxY:Double, minZ:Int, maxZ:Int): Boolean =
  {
    if ( (x >= minX) && (x <= maxX) && (y >= minY) && (y <= maxY) && (z >= minZ) && (z <= maxZ) ){
      return true
    }
    return false
  }

  def X (sigmaXj:Long, numCells:Double ):Double= {

    val x = sigmaXj / numCells  // X Bar = (sigma Xj)/ n
    return x
  }

  def BigS(sigmaSquareXj2:Double, numCells:Double, X:Double): Double = {
    val bigS = math.sqrt((sigmaSquareXj2 / numCells) - (X * X))
    return bigS
  }


  def CountNeighbours(minX: Int, minY: Int, minZ: Int, maxX: Int, maxY: Int, maxZ: Int, totalX: Int, totalY: Int, totalZ: Int): Int =
  {
    //Each cell has 26 neighbours, 6 share a common face, 12 share a edge, 8 share a corner
    var neighboursTotal = 0;

    //check if inputX is within range
    if (totalX == minX || totalX == maxX) {
      neighboursTotal += 1;
    }
    //check if inputY is within range
    if (totalY == minY || totalY == maxY) {
      neighboursTotal += 1;
    }
    //check if inputZ is within range
    if (totalZ == minZ || totalZ == maxZ) {
      neighboursTotal += 1;
    }

    val total = neighboursTotal match {
      case 0  => 27 //26
      case 1  => 18 //17
      case 2  => 12 //11
      case 3  => 8 //7
    }
    return total

  }

  def GIScore( numcells: Int, X:Double, sd: Double, totalNeighbours: Int, sumAllNeighboursPoints: Int): Double ={
    val numerator = sumAllNeighboursPoints.toDouble - (X * totalNeighbours.toDouble)

    val denominator = sd * math.sqrt(((numcells.toDouble * totalNeighbours.toDouble) - (totalNeighbours.toDouble * totalNeighbours.toDouble)) / (numcells.toDouble-1.0))
    return (numerator/denominator).toDouble
  }
}