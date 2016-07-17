package com.github.levyfan.reid.bow;

import com.github.levyfan.reid.BowImage;
import com.google.common.collect.TreeMultimap;
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

    public static Pair<int[], double[]> vote(double[] feature, List<double[]> codebook, int K, double sigma) {
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

    public Bow(int K, double sigma) {
        this.K = K;
        this.sigma = sigma;
    }

    public List<double[]> bow(BowImage bowImage, List<double[]> feature, List<double[]> codebook) {
        Strip[] strips = bowImage.strip4;
        return bow(strips, feature, codebook);
    }

    public List<double[]> bow(Strip[] strip, List<double[]> feature, List<double[]> codebook) {
        int[] tf = new int[codebook.size()];
        List<int[]> words = new ArrayList<>(feature.size());
        List<double[]> wwords = new ArrayList<>(feature.size());
        for (double[] point : feature) {
            Pair<int[], double[]> pair = vote(point, codebook, K, sigma);
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

        List<double[]> hist = new ArrayList<>(strip.length);
        for (Strip aStrip : strip) {
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
        return hist;
    }
}
