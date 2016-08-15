package com.github.levyfan.reid.util;

import com.google.common.collect.Iterables;
import com.google.common.primitives.Doubles;
import com.jmatio.types.MLDouble;
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

    public static void inplaceAdd(RealMatrix m, RealMatrix tmp) {
        for (int row = 0; row < m.getRowDimension(); row++) {
            for (int col = 0; col < m.getColumnDimension(); col++) {
                m.setEntry(row, col, m.getEntry(row, col) + tmp.getEntry(row, col));
            }
        }
    }
}
