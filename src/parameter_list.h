#ifndef PARAMETER_LIST_H
#define PARAMETER_LIST_H

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include "Parser.h"
#include <petsc.h>
#include <src/my_p4est_utils.h>

class parameter_list_t
{
  struct dbls_t {
    std::vector<double *>    ptrs;
    std::vector<std::string> keys;
    std::vector<std::string> desc;
  } dbls;
  struct ints_t {
    std::vector<int *>       ptrs;
    std::vector<std::string> keys;
    std::vector<std::string> desc;
  } ints;
  struct blns_t {
    std::vector<bool *>      ptrs;
    std::vector<std::string> keys;
    std::vector<std::string> desc;
  } blns;
  struct strs_t {
    std::vector<std::string *> ptrs;
    std::vector<std::string>   keys;
    std::vector<std::string>   desc;
  } strs;
  struct chrs_t {
    std::vector<char **>     ptrs;
    std::vector<std::string> keys;
    std::vector<std::string> desc;
  } chrs;
  struct bcs_t {
    std::vector<BoundaryConditionType *> ptrs;
    std::vector<std::string> keys;
    std::vector<std::string> desc;
  } bcs;
public:
  parameter_list_t();

  void add(double &parameter, const std::string& key, const std::string& description)
  {
    dbls.ptrs.push_back(&parameter);
    dbls.keys.push_back(key);
    dbls.desc.push_back(description);
  }

  void add(int &parameter, const std::string& key, const std::string& description)
  {
    ints.ptrs.push_back(&parameter);
    ints.keys.push_back(key);
    ints.desc.push_back(description);
  }

  void add(bool &parameter, const std::string& key, const std::string& description)
  {
    blns.ptrs.push_back(&parameter);
    blns.keys.push_back(key);
    blns.desc.push_back(description);
  }

  void add(std::string &parameter, const std::string& key, const std::string& description)
  {
    strs.ptrs.push_back(&parameter);
    strs.keys.push_back(key);
    strs.desc.push_back(description);
  }

  void add(char *&parameter, const std::string& key, const std::string& description)
  {
    chrs.ptrs.push_back(&parameter);
    chrs.keys.push_back(key);
    chrs.desc.push_back(description);
  }

  void add(BoundaryConditionType &parameter, const std::string& key, const std::string& description)
  {
    bcs.ptrs.push_back(&parameter);
    bcs.keys.push_back(key);
    bcs.desc.push_back(description);
  }

  void initialize_parser(cmdParser &cmd)
  {
    for (size_t i = 0; i < dbls.ptrs.size(); ++i) cmd.add_option(dbls.keys[i], dbls.desc[i]);
    for (size_t i = 0; i < ints.ptrs.size(); ++i) cmd.add_option(ints.keys[i], ints.desc[i]);
    for (size_t i = 0; i < blns.ptrs.size(); ++i) cmd.add_option(blns.keys[i], blns.desc[i]);
    for (size_t i = 0; i < strs.ptrs.size(); ++i) cmd.add_option(strs.keys[i], strs.desc[i]);
    for (size_t i = 0; i < chrs.ptrs.size(); ++i) cmd.add_option(chrs.keys[i], chrs.desc[i]);
    for (size_t i = 0; i < bcs .ptrs.size(); ++i) cmd.add_option(bcs .keys[i], bcs .desc[i]);
  }

  void get_all(cmdParser &cmd)
  {
    for (size_t i = 0; i < dbls.ptrs.size(); ++i) (*dbls.ptrs[i]) = cmd.get(dbls.keys[i], *dbls.ptrs[i]);
    for (size_t i = 0; i < ints.ptrs.size(); ++i) (*ints.ptrs[i]) = cmd.get(ints.keys[i], *ints.ptrs[i]);
    for (size_t i = 0; i < blns.ptrs.size(); ++i) (*blns.ptrs[i]) = cmd.get(blns.keys[i], *blns.ptrs[i]);
    for (size_t i = 0; i < strs.ptrs.size(); ++i) (*strs.ptrs[i]) = cmd.get(strs.keys[i], *strs.ptrs[i]);
    for (size_t i = 0; i < chrs.ptrs.size(); ++i) (*chrs.ptrs[i]) = cmd.get(chrs.keys[i], *chrs.ptrs[i]);
    for (size_t i = 0; i < bcs .ptrs.size(); ++i) (*bcs .ptrs[i]) = cmd.get(bcs .keys[i], *bcs .ptrs[i]);
  }

  void print_all()
  {
    PetscPrintf(MPI_COMM_WORLD, "\n -------------------== CASL List of all Parameters ==------------------- \n");
    for (size_t i = 0; i < dbls.ptrs.size(); ++i) std::cout << dbls.keys[i] << ": " << (*dbls.ptrs[i]) << std::endl;
    for (size_t i = 0; i < ints.ptrs.size(); ++i) std::cout << ints.keys[i] << ": " << (*ints.ptrs[i]) << std::endl;
    for (size_t i = 0; i < blns.ptrs.size(); ++i) std::cout << blns.keys[i] << ": " << (*blns.ptrs[i]) << std::endl;
    for (size_t i = 0; i < strs.ptrs.size(); ++i) std::cout << strs.keys[i] << ": " << (*strs.ptrs[i]) << std::endl;
    for (size_t i = 0; i < chrs.ptrs.size(); ++i) std::cout << chrs.keys[i] << ": " << (*chrs.ptrs[i]) << std::endl;
    for (size_t i = 0; i < bcs .ptrs.size(); ++i) std::cout << bcs .keys[i] << ": " << (*bcs .ptrs[i]) << std::endl;

    if (dbls.ptrs.size() == ints.ptrs.size() == blns.ptrs.size() == strs.ptrs.size() == chrs.ptrs.size() == bcs.ptrs.size() == 0) PetscPrintf(MPI_COMM_WORLD, "  NONE \n");
    PetscPrintf(MPI_COMM_WORLD," -----------------------------------------------------------------------\n");
  }

  void save_all(const std::string & output)
  {
    std::ofstream out(output);
    if (out.is_open())
    {
      for (size_t i = 0; i < dbls.ptrs.size(); ++i) out << dbls.keys[i] << ": " << (*dbls.ptrs[i]) << std::endl;
      for (size_t i = 0; i < ints.ptrs.size(); ++i) out << ints.keys[i] << ": " << (*ints.ptrs[i]) << std::endl;
      for (size_t i = 0; i < blns.ptrs.size(); ++i) out << blns.keys[i] << ": " << (*blns.ptrs[i]) << std::endl;
      for (size_t i = 0; i < strs.ptrs.size(); ++i) out << strs.keys[i] << ": " << (*strs.ptrs[i]) << std::endl;
      for (size_t i = 0; i < chrs.ptrs.size(); ++i) out << chrs.keys[i] << ": " << (*chrs.ptrs[i]) << std::endl;
      for (size_t i = 0; i < bcs .ptrs.size(); ++i) out << bcs .keys[i] << ": " << (*bcs .ptrs[i]) << std::endl;
      out.close();
    }
    else throw std::invalid_argument("Unable to open file " + output + "\n");
  }

    void generate_bash_file(const std::string & casl_directory, const std::string & example_name, const std::string & bash_file_name)
    {
        std::string file_name = casl_directory + "casl_p4est/examples/" + example_name + "/bash_runs/" + bash_file_name;
        std::ofstream out(file_name);
        if (out.is_open())
        {
            out << "#!bash/sh\n\n";
            out << "export RELEASE_OR_DEBUG=release\n";
            out << "export EXAMPLE_NAME=" << example_name << "\n";
            out << "export OUTPUT_DIR=" << casl_directory << "simulation_output/$EXAMPLE_NAME\n";
            out << "export EXECUTABLE_DIR=" << casl_directory << "built_examples/$EXAMPLE_NAME\n";
            out << "echo \"EXECUTABLE_DIR=\"$EXECUTABLE_DIR\n"
                   "echo \"OUTPUT_DIR=\"$OUTPUT_DIR\n"
                   "\n"
                   "# Output directories:-------------------------------------------------\n"
                   "\n"
                   "export OUT_DIR_VTK=$OUTPUT_DIR\n"
                   "export OUT_DIR_FILES=$OUTPUT_DIR\n"
                   "\n"
                   "# Save and load information:------------------------------------------\n"
                   "export LOAD_STATE_PATH=$OUTPUT_DIR\n"
                   "export OUT_DIR_SAVE_STATE=$OUTPUT_DIR\n\n";
            out << "# Parameters:-------------------------------------------------\n";

            for (size_t i = 0; i < dbls.ptrs.size(); ++i) {
                std::string parameter_name              = dbls.keys[i]; for (auto & c: parameter_name) c = toupper(c);
                double      parameter_value             = (*dbls.ptrs[i]);
                std::string parameter_description       = (dbls.desc[i]);
                std::string parameter_short_description = parameter_description.substr(0, parameter_description.find('\n'));
                out << "export " << parameter_name << "=" << parameter_value << " # " << parameter_short_description << std::endl;
            }
            for (size_t i = 0; i < ints.ptrs.size(); ++i) {
                std::string parameter_name              = ints.keys[i]; for (auto & c: parameter_name) c = toupper(c);
                int         parameter_value             = (*ints.ptrs[i]);
                std::string parameter_description       = (ints.desc[i]);
                std::string parameter_short_description = parameter_description.substr(0, parameter_description.find('\n'));
                out << "export " << parameter_name << "=" << parameter_value << " # " << parameter_short_description << std::endl;
            }
            for (size_t i = 0; i < blns.ptrs.size(); ++i){
                std::string parameter_name              = blns.keys[i]; for (auto & c: parameter_name) c = toupper(c);
                bool        parameter_value             = (*blns.ptrs[i]);
                std::string parameter_description       = (blns.desc[i]);
                std::string parameter_short_description = parameter_description.substr(0, parameter_description.find('\n'));
                out << "export " << parameter_name << "=" << parameter_value << " # " << parameter_short_description << std::endl;
            }
            for (size_t i = 0; i < strs.ptrs.size(); ++i) {
                std::string parameter_name              = strs.keys[i]; for (auto & c: parameter_name) c = toupper(c);
                std::string parameter_value             = (*strs.ptrs[i]);
                std::string parameter_description       = (strs.desc[i]);
                std::string parameter_short_description = parameter_description.substr(0, parameter_description.find('\n'));
                out << "export " << parameter_name << "=" << parameter_value << " # " << parameter_short_description << std::endl;
            }
            for (size_t i = 0; i < chrs.ptrs.size(); ++i) {
                std::string parameter_name              = chrs.keys[i]; for (auto & c: parameter_name) c = toupper(c);
                char*       parameter_value             = (*chrs.ptrs[i]);
                std::string parameter_description       = (chrs.desc[i]);
                std::string parameter_short_description = parameter_description.substr(0, parameter_description.find('\n'));
                out << "export " << parameter_name << "=" << parameter_value << " # " << parameter_short_description << std::endl;
            }
            for (size_t i = 0; i < bcs .ptrs.size(); ++i) {
                std::string                 parameter_name              = bcs.keys[i]; for (auto & c: parameter_name) c = toupper(c);
                BoundaryConditionType       parameter_value             = (*bcs.ptrs[i]);
                std::string                 parameter_description       = (bcs.desc[i]);
                std::string                 parameter_short_description = parameter_description.substr(0, parameter_description.find('\n'));
                out << "export " << parameter_name << "=" << parameter_value << " # " << parameter_short_description << std::endl;
            }

            out << "\n\n#create the output directory if it does not already exist - if it already exists, we overwrite the data.\n";
            out << "mkdir $OUTPUT_DIR\n";
            out << "\n #name the logfile:\n";
            out << "export LOGNAME=logfile_\"lmin\"$LMIN\"lmax\"$LMAX\"_numsplits_\"$NUM_SPLITS";
            out << "\n\n";
            out << "# Run the example with the above parameters:-------------------------------------------------\n";
            out << "mpirun -n 5 $EXECUTABLE_DIR/cmake_build_$RELEASE_OR_DEBUG/" << example_name << " ";
            for (size_t i = 0; i < dbls.ptrs.size(); ++i) {
                std::string parameter_name  = dbls.keys[i];
                double      parameter_value = (*dbls.ptrs[i]);
                out << "-" << parameter_name;
                for (auto & c: parameter_name) c = toupper(c);
                out << " $" << parameter_name << " ";
            }
            for (size_t i = 0; i < ints.ptrs.size(); ++i) {
                std::string parameter_name  = ints.keys[i];
                int         parameter_value = (*ints.ptrs[i]);
                out << "-" << parameter_name;
                for (auto & c: parameter_name) c = toupper(c);
                out << " $" << parameter_name << " ";
            }
            for (size_t i = 0; i < blns.ptrs.size(); ++i){
                std::string parameter_name  = blns.keys[i];
                bool        parameter_value = (*blns.ptrs[i]);
                out << "-" << parameter_name;
                for (auto & c: parameter_name) c = toupper(c);
                out << " $" << parameter_name << " ";
            }
            for (size_t i = 0; i < strs.ptrs.size(); ++i) {
                std::string parameter_name  = strs.keys[i];
                std::string parameter_value = (*strs.ptrs[i]);
                out << "-" << parameter_name;
                for (auto & c: parameter_name) c = toupper(c);
                out << " $" << parameter_name << " ";
            }
            for (size_t i = 0; i < chrs.ptrs.size(); ++i) {
                std::string parameter_name  = chrs.keys[i];
                char*       parameter_value = (*chrs.ptrs[i]);
                out << "-" << parameter_name;
                for (auto & c: parameter_name) c = toupper(c);
                out << " $" << parameter_name << " ";
            }
            for (size_t i = 0; i < bcs .ptrs.size(); ++i) {
                std::string           parameter_name  = bcs.keys[i];
                BoundaryConditionType parameter_value = (*bcs.ptrs[i]);
                out << "-" << parameter_name;
                for (auto & c: parameter_name) c = toupper(c);
                out << " $" << parameter_name << " ";
            }
            out << " | tee -a $OUTPUT_DIR/$LOGNAME";
            out.close();
        }
        else throw std::invalid_argument("Unable to open file " + file_name + "\n");
    }
};

class add_to_list
{
public:
  add_to_list(parameter_list_t &list, double &parameter, const std::string& key, const std::string& description) { list.add(parameter, key, description); }
  add_to_list(parameter_list_t &list, bool &parameter, const std::string& key, const std::string& description) { list.add(parameter, key, description); }
  add_to_list(parameter_list_t &list, int &parameter, const std::string& key, const std::string& description) { list.add(parameter, key, description); }
  add_to_list(parameter_list_t &list, std::string &parameter, const std::string& key, const std::string& description) { list.add(parameter, key, description); }
  add_to_list(parameter_list_t &list, char *&parameter, const std::string& key, const std::string& description) { list.add(parameter, key, description); }
  add_to_list(parameter_list_t &list, BoundaryConditionType &parameter, const std::string& key, const std::string& description) { list.add(parameter, key, description); }
};

#define ICAT(A,B) A ## B
#define CAT(A,B) ICAT(A,B)

#define DEFINE_PARAMETER(list, type, var, value, description) type var = value; add_to_list CAT(adding_,var) (list, var, #var, description)
#define ADD_TO_LIST(list, var, description) static add_to_list CAT(adding_,var) (list, var, #var, description)


/*!
 * \brief Classes param_list_t and param_t provides the user with a fast and convenient way to define and manage parameters,
 * TODO: add description
 */
class param_base_t
{
public:
  param_base_t(const std::string& key, const std::string& description)
    : key(key), description(description) {}
  std::string key;
  std::string description;
  virtual void set_from_cmd(cmdParser &cmd)=0;
  virtual std::string print_value()=0;
};

class param_list_t
{
public:
  std::vector<param_base_t *> list;

  void initialize_parser(cmdParser &cmd)
  {
    for (size_t i = 0; i < list.size(); ++i)
    {
      cmd.add_option(list[i]->key, list[i]->description);
    }
  }

  void set_from_cmd_all(cmdParser &cmd)
  {
    for (size_t i = 0; i < list.size(); ++i)
    {
      list[i]->set_from_cmd(cmd);
    }
  }

  void print_all(bool align=false)
  {
    size_t length = 0;

    if (align)
    {
      for (size_t i = 0; i < list.size(); ++i)
        if (list[i]->key.size() > length)
          length = list[i]->key.size();


      for (size_t i = 0; i < list.size(); ++i)
      {
        std::cout << list[i]->key << ":" << std::string(length-list[i]->key.size()+1, ' ') << list[i]->print_value() << "\n";
      }
    }
    else
    {
      for (size_t i = 0; i < list.size(); ++i)
      {
        std::cout << list[i]->key << ": " << list[i]->print_value() << "\n";
      }
    }
  }

  void save_all(const char output[], bool align=false)
  {
    std::ofstream out(output);
    if (out.is_open())
    {
      size_t length = 0;

      if (align)
      {
        for (size_t i = 0; i < list.size(); ++i)
          if (list[i]->key.size() > length)
            length = list[i]->key.size();


        for (size_t i = 0; i < list.size(); ++i)
        {
          out << "-" << list[i]->key << " " << std::string(length-list[i]->key.size()+1, ' ') << list[i]->print_value() << "\n";
        }
      }
      else
      {
        for (size_t i = 0; i < list.size(); ++i)
        {
          out << "-" << list[i]->key << " " << list[i]->print_value() << "\n";
        }
      }

      out.close();
    }
    else throw std::invalid_argument("Unable to open file\n");
  }
};

template <typename T>
class param_t : public param_base_t
{
public:
  param_t(T value, const std::string& key, const std::string& description)
    : param_base_t(key, description), val(value) {}

  param_t(T value, param_list_t &list, const std::string& key, const std::string& description)
    : param_base_t(key, description), val(value)
  {
    list.list.push_back(this);
  }

  param_t(param_list_t &list, T value, const std::string& key, const std::string& description)
    : param_base_t(key, description), val(value)
  {
    list.list.push_back(this);
  }

  T val;
  inline T operator()() { return val; }
  inline void set(T value) { this->val = value; }
  inline void get() { return val; }
  void set_from_cmd(cmdParser &cmd) { val = cmd.get(key, val); }
  std::string print_value()
  {
    std::ostringstream result;
    result << val;
    return result.str();
  }
};


#endif // PARAMETER_LIST_H
