package com.github.levyfan.reid.bow;

import com.github.levyfan.reid.App;
import com.github.levyfan.reid.BowImage;
import com.github.levyfan.reid.feature.Feature;
import com.github.levyfan.reid.feature.FeatureManager;
import com.google.common.collect.Sets;

import java.util.EnumSet;
import java.util.Set;

/**
 * @author fanliwen
 */
public class BowManager {

    private Bow bow;
    private FeatureManager featureManager;
    private boolean wordLevel;

    private Set<Feature.Type> types = Sets.newHashSet(App.types);

    public BowManager(Bow bow, FeatureManager featureManager, boolean wordLevel) {
        this.bow = bow;
        this.featureManager = featureManager;
        this.wordLevel = wordLevel;
    }

    public Bow getBow() {
        return bow;
    }

    public void bow(BowImage bowImage) {
        featureManager.feature(bowImage);

        for (Feature.Type type : EnumSet.allOf(Feature.Type.class)) {
            if (type == Feature.Type.ALL) {
                if (wordLevel) {
                    bow.bow(bowImage, Feature.Type.ALL);
                }
            } else if (types.contains(type)) {
                bow.bow(bowImage, type);
            } else {
//                bow.bow(bowImage, type);
            }
        }

        bowImage.sp4 = null;
        bowImage.strip4 = null;
    }

    @Override
    public String toString() {
        return "BowManager{" +
                "bow=" + bow +
                ", featureManager=" + featureManager +
                ", wordLevel=" + wordLevel +
                '}';
    }
}
