package com.github.levyfan.reid.ml;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.bow.Strip;
import com.github.levyfan.reid.feature.Feature;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import org.apache.commons.math3.linear.*;
import org.apache.commons.math3.util.Pair;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * @author fanliwen
 */
public class KissMe {

    private static final double ZERO = 10e-10;
    private static final double EPS = 10e-6;

    private static final int WORKER_NUM = 4;
    private ExecutorService executorService = Executors.newWorkStealingPool(WORKER_NUM);

    public RealMatrix apply(List<BowImage> bowImages, Feature.Type type) {
        Pair<RealMatrix, RealMatrix> pair = pairPositiveNegative(bowImages, type);
        RealMatrix positive = pair.getFirst();
        RealMatrix negative = pair.getSecond();

        RealMatrix M = inv(positive).subtract(inv(negative));
        return validateCovMatrix(M);
    }

    public RealMatrix apply(List<BowImage> bowImages) throws ExecutionException, InterruptedException {
        Pair<RealMatrix, RealMatrix> pair = pairPositiveNegative(bowImages);
        RealMatrix positive = pair.getFirst();
        RealMatrix negative = pair.getSecond();

        RealMatrix M = inv(positive).subtract(inv(negative));
        return validateCovMatrix(M);
    }

    Pair<RealMatrix, RealMatrix> pairPositiveNegative(List<BowImage> bowImages, Feature.Type type) {
        int length = bowImages.get(0).sp4[0].features.get(type).length;

        Multimap<String, Integer> idMap = ArrayListMultimap.create();
        IntStream.range(0, bowImages.size()).forEach(i -> idMap.put(bowImages.get(i).id, i));

        List<Object[]> objects = IntStream.range(0, bowImages.size()).parallel().mapToObj(i -> {
            RealMatrix positive = MatrixUtils.createRealMatrix(length, length);
            RealMatrix negative = MatrixUtils.createRealMatrix(length, length);

            BowImage x = bowImages.get(i);
            System.out.println("kissme: " + x.id);

            // positive
            long countPositive = idMap.get(x.id).stream().mapToLong(j -> {
                BowImage y = bowImages.get(j);
                if (Objects.equals(x.id, y.id) && j > i && !Objects.equals(x.cam, y.cam)) {
                    Pair<RealMatrix, Long> pair = km(x, y, type);
                    com.github.levyfan.reid.util.MatrixUtils.inplaceAdd(positive, pair.getFirst());
                    return pair.getSecond();
                } else {
                    return 0;
                }
            }).sum();

            // negative
            long countNegative = ThreadLocalRandom.current().ints(5, 0, bowImages.size()).mapToLong(j -> {
                BowImage y = bowImages.get(j);
                if (!Objects.equals(x.id, y.id) && !Objects.equals(x.cam, y.cam)) {
                    Pair<RealMatrix, Long> pair = km(x, y, type);
                    com.github.levyfan.reid.util.MatrixUtils.inplaceAdd(negative, pair.getFirst());
                    return pair.getSecond();
                } else {
                    return 0;
                }
            }).sum();

            return new Object[] {positive, negative, countPositive, countNegative};
        }).collect(Collectors.toList());

        RealMatrix positive = MatrixUtils.createRealMatrix(length, length);
        RealMatrix negative = MatrixUtils.createRealMatrix(length, length);
        long countPositive = 0;
        long countNegative = 0;
        for (Object[] obj : objects) {
            com.github.levyfan.reid.util.MatrixUtils.inplaceAdd(positive, (RealMatrix) obj[0]);
            countPositive += (long) obj[2];
            com.github.levyfan.reid.util.MatrixUtils.inplaceAdd(negative, (RealMatrix) obj[1]);
            countNegative += (long) obj[3];
        }

        positive = positive.scalarMultiply(1.0/(double) countPositive);
        negative = negative.scalarMultiply(1.0/(double) countNegative);

        return Pair.create(positive, negative);
    }

    Pair<RealMatrix, RealMatrix> pairPositiveNegative(List<BowImage> bowImages) throws ExecutionException, InterruptedException {
        int length = bowImages.get(0).hist.get(Feature.Type.ALL).length;

        Multimap<String, Integer> idMap = ArrayListMultimap.create();
        IntStream.range(0, bowImages.size()).forEach(i -> idMap.put(bowImages.get(i).id, i));

        List<List<BowImage>> input = Lists.partition(
                bowImages, bowImages.size() / WORKER_NUM);

        List<Callable<Object[]>> tasks = input.stream()
                .map(list -> new PairPN(length, Lists.newArrayList(list), bowImages, idMap))
                .collect(Collectors.toList());

        System.out.println("workers=" + WORKER_NUM);
        System.out.println("task_nums=" + tasks.size());

        List<Future<Object[]>> output = executorService.invokeAll(tasks);

        RealMatrix positive = null;
        RealMatrix negative = null;
        long countPositive = 0;
        long countNegative = 0;

        for (Future<Object[]> future : output) {
            Object[] result = future.get();

            if (positive == null) {
                positive = (RealMatrix) result[0];
            } else {
                com.github.levyfan.reid.util.MatrixUtils.inplaceAdd(positive, (RealMatrix) result[0]);
            }

            if (negative == null) {
                negative = (RealMatrix) result[1];
            } else {
                com.github.levyfan.reid.util.MatrixUtils.inplaceAdd(negative, (RealMatrix) result[1]);
            }

            countPositive += ((Pair<Long, Long>) result[2]).getFirst();
            countNegative += ((Pair<Long, Long>) result[2]).getSecond();
        }

        positive = positive.scalarMultiply(1.0/(double) countPositive);
        negative = negative.scalarMultiply(1.0/(double) countNegative);
        return Pair.create(positive, negative);
    }

    static RealMatrix validateCovMatrix(RealMatrix sig) {
        try {
            new CholeskyDecomposition(sig, 1.0e-5, 1.0e-10);
            return sig;
        } catch (NonPositiveDefiniteMatrixException | NonSymmetricMatrixException e) {
            if (e instanceof NonSymmetricMatrixException) {
                NonSymmetricMatrixException ee = (NonSymmetricMatrixException) e;
                System.out.println("(" + ee.getRow() + "," + ee.getColumn() + ")="
                        + sig.getEntry(ee.getRow(), ee.getColumn()));
                System.out.println("(" + ee.getColumn() + "," + ee.getRow() + ")="
                        + sig.getEntry(ee.getColumn(), ee.getRow()));
            }

            EigenDecomposition eigen = new EigenDecomposition(sig);
            RealMatrix v = eigen.getV();
            RealMatrix d = eigen.getD().copy();

            for (int n = 0; n < d.getColumnDimension(); n ++) {
                if (d.getEntry(n, n) < ZERO) {
                    d.setEntry(n, n, EPS);
                }
            }

            sig = v.multiply(d).multiply(v.transpose());
            try {
                new CholeskyDecomposition(sig,
                        CholeskyDecomposition.DEFAULT_ABSOLUTE_POSITIVITY_THRESHOLD,
                        CholeskyDecomposition.DEFAULT_ABSOLUTE_POSITIVITY_THRESHOLD);
            } catch (Exception ee) {
                ee.printStackTrace();
            }
            return sig;
        }
    }

    static RealMatrix inv(RealMatrix matrix) {
        try {
            return new LUDecomposition(matrix).getSolver().getInverse();
        } catch (SingularMatrixException e) {
            e.printStackTrace();
            return new SingularValueDecomposition(matrix).getSolver().getInverse();
        }
    }

    private static Pair<RealMatrix, Long> km(BowImage i, BowImage j, Feature.Type type) {
        int length = i.sp4[0].features.get(type).length;
        RealMatrix matrix = MatrixUtils.createRealMatrix(length, length);

        long count = 0;

        for (int nstrip = 0; nstrip < i.strip4.length; nstrip ++) {
            Strip iStrip = i.strip4[nstrip];
            Strip jStrip = j.strip4[nstrip];

            for (int iSuperPixel : iStrip.superPixels) {
                for (int jSuperPixel : jStrip.superPixels) {
                    ArrayRealVector iFeature = new ArrayRealVector(i.sp4[iSuperPixel].features.get(type), false);
                    ArrayRealVector jFeature = new ArrayRealVector(j.sp4[jSuperPixel].features.get(type), false);
                    ArrayRealVector delta = iFeature.subtract(jFeature);

                    com.github.levyfan.reid.util.MatrixUtils.inplaceAdd(matrix, delta.outerProduct(delta));
                    count++;
                }
            }
        }
        return Pair.create(matrix, count);
    }

    private static RealMatrix km(BowImage i, BowImage j) {
        ArrayRealVector iHist = new ArrayRealVector(i.hist.get(Feature.Type.ALL), false);
        ArrayRealVector jHist = new ArrayRealVector(j.hist.get(Feature.Type.ALL), false);
        ArrayRealVector delta = iHist.subtract(jHist);
        return delta.outerProduct(delta);
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
