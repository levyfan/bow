package com.github.levyfan.reid.util;

import com.google.common.collect.Iterables;
import com.google.common.primitives.Doubles;
import com.jmatio.types.MLDouble;
import com.jmatio.types.MLNumericArray;
import org.apache.commons.math3.linear.RealMatrix;

import java.util.ArrayList;
import java.util.List;

/**
 * @author fanliwen
 */
public class MatrixUtils {

    public static MLDouble to(String name, Iterable<double[]> iterable) {
        int length = Iterables.getFirst(iterable, null).length;

        List<Double> list = new ArrayList<>(Iterables.size(iterable) * length);
        for (double[] doubles : iterable) {
            list.addAll(Doubles.asList(doubles));
        }

        return new MLDouble(name, list.toArray(new Double[0]), length);
    }

    public static RealMatrix from(MLNumericArray mlNumericArray) {
        RealMatrix matrix = org.apache.commons.math3.linear.MatrixUtils.createRealMatrix(
                mlNumericArray.getM(), mlNumericArray.getN());
        for (int m = 0; m < mlNumericArray.getM(); m ++) {
            for (int n = 0; n < mlNumericArray.getN(); n ++) {
                matrix.setEntry(m, n, mlNumericArray.getReal(m, n).doubleValue());
            }
        }
        return matrix;
    }

    public static void inplaceAdd(RealMatrix m, RealMatrix tmp) {
        for (int row = 0; row < m.getRowDimension(); row++) {
            for (int col = 0; col < m.getColumnDimension(); col++) {
                m.setEntry(row, col, m.getEntry(row, col) + tmp.getEntry(row, col));
            }
        }
    }
}
