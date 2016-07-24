package com.github.levyfan.reid.util;

import com.google.common.collect.Iterables;
import com.google.common.primitives.Doubles;
import com.jmatio.types.MLDouble;

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
}
