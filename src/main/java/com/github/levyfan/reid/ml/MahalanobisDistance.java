package com.github.levyfan.reid.ml;

import org.apache.commons.math3.exception.DimensionMismatchException;
import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.ml.distance.DistanceMeasure;

/**
 * @author fanliwen
 */
public class MahalanobisDistance implements DistanceMeasure {

    private RealMatrix M;

    public MahalanobisDistance(RealMatrix M) {
        this.M = M;
    }

    @Override
    public double compute(double[] a, double[] b) throws DimensionMismatchException {
        ArrayRealVector va = new ArrayRealVector(a, false);
        ArrayRealVector vb = new ArrayRealVector(b, false);
        ArrayRealVector delta = va.subtract(vb);

        return Math.sqrt(M.preMultiply(delta).dotProduct(delta));
    }

    @Override
    public String toString() {
        return "MahalanobisDistance{}";
    }
}
