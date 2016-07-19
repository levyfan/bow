package com.github.levyfan.reid.bow;

import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.sp.SuperPixel;
import com.google.common.collect.TreeMultimap;
import com.google.common.primitives.Doubles;
import org.apache.commons.math3.util.MathArrays;
import org.apache.commons.math3.util.Pair;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

/**
 * @author fanliwen
 */
public class Bow {

    private static Pair<int[], double[]> vote(double[] feature, List<double[]> codebook, int K, double sigma) {
        TreeMultimap<Double, Integer> multimap = TreeMultimap.create();
        for (int i = 0; i < codebook.size(); i++) {
            multimap.put(MathArrays.distance(feature, codebook.get(i)), i);
        }

        Iterator<Map.Entry<Double, Integer>> iterator = multimap.entries().iterator();
        int[] words = new int[K];
        double[] wWords = new double[K];
        for (int k = 0; k < K; k++) {
            Map.Entry<Double, Integer> entry = iterator.next();
            words[k] = entry.getValue();
            wWords[k] = Math.exp(-entry.getKey() / (sigma*sigma));
        }
        return Pair.create(words, wWords);
    }

    private int K;
    private double sigma;
    private Map<Feature.Type, List<double[]>> codebooks;

    public Bow(int K, double sigma, Map<Feature.Type, List<double[]>> codebooks) {
        this.K = K;
        this.sigma = sigma;
        this.codebooks = codebooks;
    }

    void bow(BowImage bowImage, Feature.Type type) {
        List<double[]> codebook = codebooks.get(type);

        int[] tf = new int[codebook.size()];
        List<int[]> words = new ArrayList<>(bowImage.sp4.length);
        List<double[]> wwords = new ArrayList<>(bowImage.sp4.length);
        for (SuperPixel superPixel : bowImage.sp4) {
            double[] feature;
            if (type == Feature.Type.ALL) {
                feature = Doubles.concat(
                        superPixel.features.get(Feature.Type.HSV),
                        superPixel.features.get(Feature.Type.CN),
                        superPixel.features.get(Feature.Type.HOG),
                        superPixel.features.get(Feature.Type.SILTP)
                );
            } else {
                feature = superPixel.features.get(type);
            }

            Pair<int[], double[]> pair = vote(feature, codebook, K, sigma);
            words.add(pair.getFirst());
            wwords.add(pair.getSecond());
            for (int word : pair.getFirst()) {
                tf[word] ++;
            }
        }
        for (int i = 0; i < tf.length; i ++) {
            if (tf[i] == 0) {
                tf[i] = 1;
            }
        }

        List<double[]> hist = new ArrayList<>(bowImage.strip4.length);
        for (Strip aStrip : bowImage.strip4) {
            double[] tempHist = new double[codebook.size()];
            for (int nsuperpixel : aStrip.superPixels) {
                int[] word = words.get(nsuperpixel);
                double[] wword = wwords.get(nsuperpixel);

                for (int i = 0; i < word.length; i++) {
                    tempHist[word[i]] += wword[i];
                }
            }

            for (int i = 0; i < tempHist.length; i++) {
                tempHist[i] = tempHist[i] / Math.sqrt(tf[i]);
            }
            hist.add(tempHist);
        }
        bowImage.hist.put(type, Doubles.concat(hist.toArray(new double[0][])));
    }
}
