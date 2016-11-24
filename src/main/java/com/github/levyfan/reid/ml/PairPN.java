package com.github.levyfan.reid.ml;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.feature.Feature;
import com.google.common.collect.Multimap;
import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.util.Pair;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.Callable;
import java.util.concurrent.ThreadLocalRandom;

/**
 * @author fanliwen
 */
class PairPN implements Callable<Object[]> {

    private int length;
    private List<BowImage> list;
    private List<BowImage> bowImages;

    private Multimap<String, Integer> idMap;

    PairPN(int length, List<BowImage> list, List<BowImage> bowImages, Multimap<String, Integer> idMap) {
        this.length = length;
        this.list = list;
        this.bowImages = bowImages;
        this.idMap = idMap;
    }

    @Override
    public Object[] call() throws Exception {
        System.out.println(Thread.currentThread().getName() + " is called");

        RealMatrix positive = new Array2DRowRealMatrix(length, length);
        RealMatrix negative = new Array2DRowRealMatrix(length, length);

        Pair<Long, Long> counter = list.stream().map(x -> {
            System.out.println("kissme: " + x.id);

            // positive
            long countPositive = idMap.get(x.id).stream().mapToLong(j -> {
                BowImage y = bowImages.get(j);
                if (Objects.equals(x.id, y.id) && !Objects.equals(x.cam, y.cam)) {
                    km(positive, x, y);
                    return 1;
                } else {
                    return 0;
                }
            }).sum();

            // negative
            long countNegative = ThreadLocalRandom.current().ints(5, 0, bowImages.size()).mapToLong(j -> {
                BowImage y = bowImages.get(j);
                if (!Objects.equals(x.id, y.id) && !Objects.equals(x.cam, y.cam)) {
                    km(negative, x, y);
                    return 1;
                } else {
                    return 0;
                }
            }).sum();

            return Pair.create(countPositive, countNegative);
        }).reduce((p1, p2) ->
                Pair.create(p1.getFirst() + p2.getFirst(), p1.getSecond() + p2.getSecond())
        ).orElse(Pair.create(0L, 0L));

        System.out.println(Thread.currentThread().getName() + " is done");
        return new Object[]{positive, negative, counter};
    }

    private static void km(RealMatrix matrix, BowImage i, BowImage j) {
        double[] iHist = i.hist.get(Feature.Type.ALL);
        double[] jHist = j.hist.get(Feature.Type.ALL);

        for (int m = 0; m < iHist.length; m ++) {
            for (int n = 0; n < jHist.length; n ++) {
                double d = (iHist[m] - jHist[m]) * (iHist[n] - jHist[n]);
                matrix.setEntry(m, n, matrix.getEntry(m, n) + d);
            }
        }
    }
}
