package com.github.levyfan.reid;

import com.github.levyfan.reid.feature.Feature;
import com.google.common.collect.ListMultimap;
import com.jmatio.io.MatFileWriter;
import com.jmatio.types.MLArray;
import com.jmatio.types.MLDouble;
import org.apache.commons.math3.util.Pair;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.net.URISyntaxException;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * @author fanliwen
 */
public class Training extends App {

    private Training() throws IOException, URISyntaxException {
        super();
    }

    private ListMultimap<Feature.Type, double[]> featureTraining(File folder) throws IOException {
        File[] files = folder.listFiles(filter);
        return featureManager.feature(files, spMethod);
    }

    private Map<Feature.Type, List<double[]>> codeBookTraining(
            File folder, ListMultimap<Feature.Type, double[]> featureMap) throws IOException {
        Map<Feature.Type, List<double[]>> books = featureMap.asMap().entrySet()
                .parallelStream()
                .map(entry -> {
                    System.out.println("codebook gen 0 " + entry.getKey());
                    List<double[]> words = codeBook.codebook_fast(entry.getValue());
                    System.out.println("codebook gen 1 " + entry.getKey());
                    return Pair.create(entry.getKey(), words);
                }).collect(Collectors.toMap(Pair::getFirst, Pair::getSecond));

        File dat = new File(folder, "codebook_slic_" + numSuperpixels + "_" + compactness + ".dat");
        try (ObjectOutputStream os = new ObjectOutputStream(new FileOutputStream(dat))) {
            os.writeObject(books);
        }
        return books;
    }

    public static void main(String[] args) throws IOException, URISyntaxException {
        Training app = new Training();

        ListMultimap<Feature.Type, double[]> featureMap = app.featureTraining(training);
        Map<Feature.Type, List<double[]>> codebookMap = app.codeBookTraining(training, featureMap);

        List<MLArray> mlArrays = codebookMap.entrySet()
                .stream()
                .map(entry -> new MLDouble(entry.getKey().name(), entry.getValue().toArray(new double[0][])))
                .collect(Collectors.toList());
        new MatFileWriter().write(
                "codebook_" + numSuperpixels + "_" + compactness + ".mat",
                mlArrays);
    }
}
