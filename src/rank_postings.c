/* Exact top-K BM25 over sorted posting lists. No pruning/approximation.
 * Formula: SQLite FTS5 unit-weight BM25, k1=1.2, b=0.75.
 * This is a CPU performance helper, not a trained model.
 */
#include <stdint.h>
#include <stddef.h>
#include <limits.h>

static int worse(double a, uint32_t ai, double b, uint32_t bi) {
    return a>b || (a==b && ai>bi);
}
static void swap(double *s,uint32_t *ids,int a,int b) {
    double d=s[a];s[a]=s[b];s[b]=d;
    uint32_t i=ids[a];ids[a]=ids[b];ids[b]=i;
}

int rank_postings(const uint32_t **lists, const uint64_t *sizes,
                  const double *idf, int terms, int split,
                  double country_idf, uint16_t country,
                  const uint32_t *lengths, const uint16_t *countries,
                  uint64_t rows, double average, int k,
                  uint32_t *out_ids, double *out_scores) {
    uint64_t positions[8]={0}; double tf[8]; int used=0;
    if(terms<0 || terms>8 || k<0 || average<=0) return -1;
    if(k==0) return 0;
    for(;;) {
        uint32_t rid=UINT32_MAX;
        for(int j=0;j<terms;j++)
            if(positions[j]<sizes[j] && lists[j][2*positions[j]]<rid)
                rid=lists[j][2*positions[j]];
        if(rid==UINT32_MAX) break;
        if((uint64_t)rid>=rows) return -2;
        int left=0,right=0;
        for(int j=0;j<terms;j++) {
            tf[j]=0;
            if(positions[j]<sizes[j] && lists[j][2*positions[j]]==rid) {
                tf[j]=(double)lists[j][2*positions[j]+1];positions[j]++;
                if(j<split)left=1;else right=1;
            }
        }
        if(countries[rid]!=country || (split>0 && !(left && right))) continue;
        double denominator=1.2*(1.0-0.75+0.75*(double)lengths[rid]/average);
        double score=country_idf*((1.0*(1.2+1.0))/(1.0+denominator));
        for(int j=0;j<terms;j++)
            score+=idf[j]*((tf[j]*(1.2+1.0))/(tf[j]+denominator));
        score=-score;
        if(used<k) {
            int cur=used++;out_ids[cur]=rid;out_scores[cur]=score;
            while(cur>0) {
                int parent=(cur-1)/2;
                if(!worse(out_scores[cur],out_ids[cur],out_scores[parent],out_ids[parent])) break;
                swap(out_scores,out_ids,cur,parent);cur=parent;
            }
        } else if(worse(out_scores[0],out_ids[0],score,rid)) {
            out_ids[0]=rid;out_scores[0]=score;
            int cur=0;
            for(;;) {
                int child=cur*2+1;
                if(child>=used)break;
                if(child+1<used && worse(out_scores[child+1],out_ids[child+1],out_scores[child],out_ids[child]))child++;
                if(!worse(out_scores[child],out_ids[child],out_scores[cur],out_ids[cur]))break;
                swap(out_scores,out_ids,cur,child);cur=child;
            }
        }
    }
    return used;
}
